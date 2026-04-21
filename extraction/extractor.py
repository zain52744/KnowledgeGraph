import json
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        return lambda f: f

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.schemas import ExtractionResult
from api.sanitization import sanitize_entity_name, sanitize_relation_type, sanitize_description

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
   

    SYSTEM_PROMPT = """You are a knowledge graph extractor. Extract all named entities and their relationships from the text. Respond ONLY with a valid JSON object. No explanation. No markdown.

The JSON must have this exact structure:
{
  "entities": [
    {
      "id": "entity_1",
      "name": "Entity Name",
      "type": "Person|Organization|Location|Concept|Other",
      "description": "Brief description"
    }
  ],
  "relations": [
    {
      "source": "Entity Name",
      "relation": "relationship type",
      "target": "Another Entity",
      "confidence": 0.95
    }
  ]
}

Rules:
- Extract ALL entities mentioned in the text
- Only include relationships that are explicitly stated or strongly implied
- Confidence should be 0.5-1.0 based on explicitness
- If no entities or relations found, return empty lists
- IDs should be sequential: entity_1, entity_2, etc."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    @observe(name="extract_knowledge")
    def extract_knowledge(self, text: str) -> ExtractionResult:
       
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return ExtractionResult(entities=[], relations=[])

        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=text),
            ]
            response = self._call_llm(messages)
            response_text = response.content.strip()

            # Parse JSON
            extraction_result = self._parse_response(response_text)

            logger.info(
                f"Extracted {len(extraction_result.entities)} entities "
                f"and {len(extraction_result.relations)} relations"
            )
            return extraction_result

        except ValueError as e:
            logger.error(f"Validation error during extraction: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
            raise RuntimeError(f"Failed to extract knowledge: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_llm(self, messages):
       
        return self.llm.invoke(messages)

    def _parse_response(self, response_text: str) -> ExtractionResult:
       
        # Remove markdown code blocks if present
        response_text = self._clean_json_response(response_text)

        # Parse JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response_text[:200]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}") from e

        # Validate with Pydantic
        try:
            result = ExtractionResult(**data)
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            raise ValueError(f"Response does not match ExtractionResult schema: {e}") from e
        
        # Sanitize extracted entities and relations
        sanitized_entities = []
        for entity in result.entities:
            try:
                name = sanitize_entity_name(entity.name, strict=False)
                etype = sanitize_entity_name(entity.type, strict=False)
                description = sanitize_description(entity.description)
                
                if name and etype:
                    entity.name = name
                    entity.type = etype
                    if description:
                        entity.description = description
                    sanitized_entities.append(entity)
                else:
                    logger.warning(f"Skipping entity: name or type became empty after sanitization")
            except Exception as e:
                logger.warning(f"Error sanitizing entity: {e}")
                continue
        
        sanitized_relations = []
        for relation in result.relations:
            try:
                source = sanitize_entity_name(relation.source, strict=False)
                target = sanitize_entity_name(relation.target, strict=False)
                rel_type = sanitize_relation_type(relation.relation, strict=False)
                
                if source and target and rel_type:
                    relation.source = source
                    relation.target = target
                    relation.relation = rel_type
                    sanitized_relations.append(relation)
                else:
                    logger.warning(f"Skipping relation: source, target, or relation became empty after sanitization")
            except Exception as e:
                logger.warning(f"Error sanitizing relation: {e}")
                continue
        
        result.entities = sanitized_entities
        result.relations = sanitized_relations
        return result

    def _clean_json_response(self, response_text: str) -> str:
        
        # Remove markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return response_text.strip()

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.extract_knowledge(text)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to extract from text {i + 1}: {e}")
                # Add empty result to maintain indexing
                results.append(ExtractionResult(entities=[], relations=[]))

        return results

