from src.nlu_processor import normalize_hinglish_terms, correct_misspelled_entity

def test_hinglish_normalization_examples():
    assert normalize_hinglish_terms("mere papa ko bukhar aur khansi hai") == "mere papa ko fever aur cough hai"
    assert normalize_hinglish_terms("मुझे sardi ho gayi hai") == "मुझे cold ho gayi hai"
    assert normalize_hinglish_terms("pet mein jalan hai") == "pet mein burning sensation hai"
    assert normalize_hinglish_terms("mere dost ko gala kharab hai") == "mere dost ko cough hai"
    assert normalize_hinglish_terms("मुझे thakan aur chakkar aa rahe hain") == "मुझे fatigue aur dizziness aa rahe hain"
    assert normalize_hinglish_terms("मुझे सिर दर्द है") == "मुझे headache है"
    assert normalize_hinglish_terms("mere dost ko pet dard hai") == "mere dost ko stomach pain hai"