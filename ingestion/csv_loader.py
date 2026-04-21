import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.sanitization import sanitize_entity_name, sanitize_relation_type, sanitize_description

logger = logging.getLogger(__name__)


class CSVLoader:
   

    REQUIRED_COLUMNS = {"source", "relation", "target"}
    OPTIONAL_COLUMNS = {"description", "weight"}

    def __init__(self):
        
        pass

    def load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
       
        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                logger.warning(f"CSV file is empty: {csv_path}")
                return []

            # Validate required columns
            missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}. "
                    f"Found: {set(df.columns)}"
                )

            # Process rows
            edges = self._process_rows(df, csv_path)
            logger.info(f"Loaded {len(edges)} edges from {csv_path}")

            return edges

        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            raise
        except ValueError as e:
            logger.error(f"Validation error in {csv_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV from {csv_path}: {e}")
            raise RuntimeError(f"Failed to parse CSV: {e}") from e

    def _process_rows(
        self, df: pd.DataFrame, source_path: str
    ) -> List[Dict[str, Any]]:
        
        edges = []

        for idx, row in df.iterrows():
            try:
                edge = self._build_edge(row, idx)
                if edge:  # Skip rows that fail validation
                    edges.append(edge)
            except Exception as e:
                logger.warning(f"Skipping row {idx + 2} in {source_path}: {e}")
                continue

        return edges

    def _build_edge(self, row: pd.Series, row_idx: int) -> Optional[Dict[str, Any]]:
       
        # Extract required fields with validation and sanitization
        source = self._extract_field(row["source"], "source", row_idx)
        relation = self._extract_field(row["relation"], "relation", row_idx)
        target = self._extract_field(row["target"], "target", row_idx)

        if not source or not relation or not target:
            return None
        
        # Sanitize entity names and relation type
        try:
            source = sanitize_entity_name(source, strict=False)
            target = sanitize_entity_name(target, strict=False)
            relation = sanitize_relation_type(relation, strict=False)
        except ValueError as e:
            logger.warning(f"Row {row_idx + 2}: Sanitization error: {e}")
            return None
        
        if not source or not relation or not target:
            logger.warning(f"Row {row_idx + 2}: Entity names or relation became empty after sanitization")
            return None

        # Build edge dictionary
        edge = {
            "source": source,
            "relation": relation,
            "target": target,
        }

        # Add optional fields
        if "description" in row:
            desc = self._extract_optional_field(row["description"])
            if desc:
                desc = sanitize_description(desc)
                if desc:
                    edge["description"] = desc

        if "weight" in row:
            weight = self._extract_weight(row["weight"], row_idx)
            if weight is not None:
                edge["weight"] = weight

        return edge

    def _extract_field(self, value: Any, field_name: str, row_idx: int) -> Optional[str]:
       
        # Check for NaN/None
        if pd.isna(value):
            raise ValueError(f"Required field '{field_name}' is empty")

        # Convert to string and strip
        value_str = str(value).strip()

        if not value_str:
            raise ValueError(f"Required field '{field_name}' is empty after stripping")

        return value_str

    def _extract_optional_field(self, value: Any) -> Optional[str]:
        
        if pd.isna(value):
            return None

        value_str = str(value).strip()
        return value_str if value_str else None

    def _extract_weight(self, value: Any, row_idx: int) -> Optional[float]:
       
        if pd.isna(value):
            return None

        try:
            weight = float(value)
            if weight < 0:
                logger.warning(
                    f"Row {row_idx + 2}: negative weight {weight}, using absolute value"
                )
                weight = abs(weight)
            return weight
        except (ValueError, TypeError):
            logger.warning(f"Row {row_idx + 2}: invalid weight value '{value}', skipping")
            return None

    def validate_csv(self, csv_path: str) -> Dict[str, Any]:
        
        errors = []

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {
                "valid": False,
                "row_count": 0,
                "columns": [],
                "missing_required": self.REQUIRED_COLUMNS,
                "errors": [str(e)],
            }

        # Check required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)

        # Check for empty rows in required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}': {null_count} null values")

        return {
            "valid": len(missing_cols) == 0 and len(errors) == 0,
            "row_count": len(df),
            "columns": list(df.columns),
            "missing_required": list(missing_cols),
            "errors": errors,
        }

