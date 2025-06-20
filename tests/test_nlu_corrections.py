import json
import pytest
from src.nlu_processor import (
    correct_misspelled_entity,
    phonetic_match,
    COMMON_MISSPELLINGS,
)

def test_correct_misspelled_entity_more():
    """Edge cases for fuzzy misspelling correction."""
    assert correct_misspelled_entity("sugar disease") == "diabetes"
    assert correct_misspelled_entity("dibetes") == "diabetes"
    assert correct_misspelled_entity("high bp") == "hypertension"
    assert correct_misspelled_entity("hypertenion") == "hypertension"
    assert correct_misspelled_entity("mygrain") == "migraine"
    assert correct_misspelled_entity("migren") == "migraine"
    assert correct_misspelled_entity("asthama") == "asthma"
    assert correct_misspelled_entity("breathing problem") == "asthma"
    assert correct_misspelled_entity("fevr") == "fever"
    assert correct_misspelled_entity("high temp") == "fever"
    assert correct_misspelled_entity("coff") == "cough"
    assert correct_misspelled_entity("caugh") == "cough"
    assert correct_misspelled_entity("runny nose") == "cold"
    assert correct_misspelled_entity("flue") == "flu"
    assert correct_misspelled_entity("alergy") == "allergy"
    assert correct_misspelled_entity("allergies") == "allergy"
    assert correct_misspelled_entity("unknownsymptom") == "unknownsymptom"

def test_phonetic_match_more():
    """Edge cases for phonetic similarity correction."""
    candidates = tuple(COMMON_MISSPELLINGS.keys())
    assert phonetic_match("dyabates", candidates) == "diabetes"
    assert phonetic_match("dibetes", candidates) == "diabetes"
    assert phonetic_match("mygrain", candidates) == "migraine"
    assert phonetic_match("migren", candidates) == "migraine"
    assert phonetic_match("azma", candidates) == "asthma"
    assert phonetic_match("asthama", candidates) == "asthma"
    assert phonetic_match("fevar", candidates) == "fever"
    assert phonetic_match("coff", candidates) == "cough"
    assert phonetic_match("caugh", candidates) == "cough"
    assert phonetic_match("flue", candidates) == "flu"
    assert phonetic_match("alergi", candidates) == "allergy"
    assert phonetic_match("kabz", candidates) == "constipation"
    assert phonetic_match("constipated", candidates) == "constipation"
    assert phonetic_match("randomsymptom", candidates) == "randomsymptom"