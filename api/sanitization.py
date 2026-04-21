import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Allowed characters: alphanumeric, spaces, hyphens, underscores, dots, apostrophes
ENTITY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9\s\-_.']+$")

# Allowed characters for filenames: alphanumeric, dots, hyphens, underscores
FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._\-]+$")

# Maximum lengths
MAX_ENTITY_NAME_LENGTH = 255
MAX_RELATION_TYPE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5000
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    
    if not filename or not isinstance(filename, str):
        raise ValueError("Filename must be a non-empty string")
    
    filename = filename.strip()
    
    # Reject path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Filename contains invalid path characters (../, ..\\, /, \\)")
    
    # Reject if too long
    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"Filename exceeds maximum length of {MAX_FILENAME_LENGTH}")
    
    # Extract just the filename (in case full path was provided)
    filename = filename.split("\\")[-1].split("/")[-1]
    
    # Validate pattern
    if not FILENAME_PATTERN.match(filename):
        raise ValueError(
            "Filename contains invalid characters. Only alphanumeric, dots, hyphens, "
            "and underscores are allowed"
        )
    
    logger.info(f"Filename validated: {filename}")
    return filename


def sanitize_entity_name(name: str, strict: bool = True) -> Optional[str]:
   
    if not name or not isinstance(name, str):
        return None
    
    name = name.strip()
    
    if not name:
        return None
    
    # Reject if too long
    if len(name) > MAX_ENTITY_NAME_LENGTH:
        if strict:
            raise ValueError(
                f"Entity name exceeds maximum length of {MAX_ENTITY_NAME_LENGTH}"
            )
        name = name[:MAX_ENTITY_NAME_LENGTH]
    
    if strict:
        # Strict validation - reject if pattern doesn't match
        if not ENTITY_NAME_PATTERN.match(name):
            raise ValueError(
                f"Entity name '{name}' contains invalid characters. "
                "Only alphanumeric, spaces, hyphens, underscores, dots, and apostrophes allowed"
            )
    else:
        # Non-strict mode - clean the string
        # Replace problematic characters with spaces, then clean up
        name = re.sub(r"[^a-zA-Z0-9\s\-_.]+", " ", name)
        name = re.sub(r"\s+", " ", name)  # Collapse multiple spaces
        name = name.strip()
        
        if not name:
            return None
    
    return name


def sanitize_relation_type(relation: str, strict: bool = True) -> Optional[str]:
   
    if not relation or not isinstance(relation, str):
        return None
    
    relation = relation.strip()
    
    if not relation:
        return None
    
    # Reject if too long
    if len(relation) > MAX_RELATION_TYPE_LENGTH:
        if strict:
            raise ValueError(
                f"Relation type exceeds maximum length of {MAX_RELATION_TYPE_LENGTH}"
            )
        relation = relation[:MAX_RELATION_TYPE_LENGTH]
    
    if strict:
        if not ENTITY_NAME_PATTERN.match(relation):
            raise ValueError(
                f"Relation type '{relation}' contains invalid characters. "
                "Only alphanumeric, spaces, hyphens, underscores, dots, and apostrophes allowed"
            )
    else:
        # Non-strict mode - clean the string
        relation = re.sub(r"[^a-zA-Z0-9\s\-_.']+", " ", relation)
        relation = re.sub(r"\s+", " ", relation)
        relation = relation.strip()
        
        if not relation:
            return None
    
    return relation


def sanitize_description(description: str) -> Optional[str]:
   
    if not description or not isinstance(description, str):
        return None
    
    description = description.strip()
    
    if not description:
        return None
    
    # Limit length
    if len(description) > MAX_DESCRIPTION_LENGTH:
        description = description[:MAX_DESCRIPTION_LENGTH]
    
    # Remove excessive whitespace but preserve line breaks
    description = re.sub(r"[ \t]+", " ", description)
    description = re.sub(r"\n\n+", "\n", description)
    
    return description


def validate_string_for_graph(
    value: str, 
    field_type: str = "entity_name", 
    strict: bool = True
) -> Optional[str]:
   
    if field_type == "entity_name":
        return sanitize_entity_name(value, strict=strict)
    elif field_type == "relation":
        return sanitize_relation_type(value, strict=strict)
    elif field_type == "description":
        return sanitize_description(value)
    else:
        raise ValueError(f"Unknown field type: {field_type}")
