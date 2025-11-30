import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from main import ChatProcessor, DataLoader

# ==========================================
# Fixtures (Mock Data)
# ==========================================

@pytest.fixture
def mock_keyword_df():
    """Creates a DataFrame mimicking keywords.xlsx"""
    data = {
        "brand": ["BrandA", "BrandA", "BrandB"],
        "product": ["Product1", "Generic", "Product1"],
        "keyword": ["alpha", "hello", "beta"],
        "required_product": [None, None, "BrandA"], # BrandB requires BrandA
        "headers": ["BrandA_Product1", "BrandA_Generic", "BrandB_Product1"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_chat_df():
    """Creates a DataFrame mimicking a chat log CSV."""
    data = {
        "MessageBody": [
            "I like alpha product",       # Matches BrandA_Product1
            "Hello there",                # Matches BrandA_Generic
            "I like alpha and beta",      # Matches BrandB (requires BrandA/alpha)
            "Just random text"            # Matches nothing
        ],
        "Source": ["file1.csv"] * 4
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_analyzer():
    """Mocks the SentimentAnalyzer."""
    analyzer = MagicMock()
    analyzer.analyze.return_value = {"sentiment": "P", "reason": "Test reason"}
    return analyzer

@pytest.fixture
def processor(mock_keyword_df, mock_analyzer):
    """Initializes ChatProcessor."""
    return ChatProcessor(mock_keyword_df, mock_analyzer)

# ==========================================
# Tests for ChatProcessor Logic
# ==========================================

def test_resolve_required_keywords(processor):
    """Test if required keywords are looked up correctly."""
    # Row 2 in mock_keyword_df is BrandB, requires "BrandA" products
    # BrandA products correspond to keyword "alpha" (from row 0)
    
    row = processor.keyword_df.iloc[2] 
    result = processor._resolve_required_keywords(row)
    
    # It should find "alpha" because BrandA matches row 0's product
    assert "alpha" in result

def test_tag_keywords_basic(processor):
    """Test basic keyword matching."""
    df = pd.DataFrame({"MessageBody": ["I like alpha"]})
    for h in processor.headers:
        df[h] = 0
        
    processor._tag_keywords(df)
    
    # "alpha" maps to BrandA_Product1
    assert df.loc[0, "BrandA_Product1"] == 1
    assert df.loc[0, "BrandA_Generic"] == 0

def test_tag_keywords_generic_exclusion(processor):
    """
    CRITICAL: Test that 'Generic' is skipped if a specific brand is found.
    """
    # "alpha" = BrandA_Product1 (Specific)
    # "hello" = BrandA_Generic (Generic)
    df = pd.DataFrame({"MessageBody": ["Hello, I like alpha"]})
    for h in processor.headers:
        df[h] = 0
        
    processor._tag_keywords(df)
    
    # Even though "Hello" is present, "alpha" is also present.
    # Logic: If specific subbrand found, skip generic.
    assert df.loc[0, "BrandA_Product1"] == 1
    assert df.loc[0, "BrandA_Generic"] == 0 

def test_tag_keywords_generic_inclusion(processor):
    """Test that 'Generic' IS tagged if no specific brand is found."""
    df = pd.DataFrame({"MessageBody": ["Hello only"]})
    for h in processor.headers:
        df[h] = 0
        
    processor._tag_keywords(df)
    
    assert df.loc[0, "BrandA_Product1"] == 0
    assert df.loc[0, "BrandA_Generic"] == 1

def test_process_folder_structure_and_types(processor, mock_chat_df):
    """
    Test the full pipeline:
    1. Tags are applied.
    2. 0s are converted to empty strings.
    3. Columns are object type.
    """
    # Mock _run_sentiment_analysis to avoid complex apply logic in this specific test
    processor._run_sentiment_analysis = MagicMock()
    
    result_df = processor.process_folder(mock_chat_df)
    
    # Check Row 0: "I like alpha product" -> BrandA_Product1 should be 1 (or "1")
    val = result_df.loc[0, "BrandA_Product1"]
    assert val == 1 or val == "1"
    
    # Check Row 0: BrandA_Generic should be empty string (not 0)
    assert result_df.loc[0, "BrandA_Generic"] == ""
    
    # Check Dtypes: Should be object (to hold both strings and empty strings)
    assert result_df["BrandA_Product1"].dtype == "O"

def test_pass_to_llm(processor):
    """Test that the LLM result is written to the row."""
    row = pd.Series({"MessageBody": "test msg"})
    header = "BrandA_Product1"
    keywords = "alpha"
    
    updated_row = processor._pass_to_llm(row, header, keywords)
    
    assert updated_row[header] == "P"
    assert updated_row["Reason"] == "Test reason"

# ==========================================
# Tests for DataLoader
# ==========================================

def test_dataloader_load_keywords():
    """Test loading keywords adds the 'headers' column."""
    loader = DataLoader()
    
    # Mock pd.read_excel
    with patch("pandas.read_excel") as mock_read:
        mock_read.return_value = pd.DataFrame({
            "brand": ["A"], "product": ["B"]
        })
        
        df = loader.load_keywords()
        
        assert "headers" in df.columns
        assert df.loc[0, "headers"] == "A_B"

def test_dataloader_load_chat_folder(tmp_path):
    """
    Test loading CSVs from a folder.
    Uses pytest's tmp_path to create real temporary files.
    """
    loader = DataLoader()
    
    # Create a dummy CSV file
    d = tmp_path / "chat_data"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("Date1,MessageBody\n2023-01-01,Hello")
    
    df = loader.load_chat_folder(d)
    
    assert not df.empty
    assert "Source" in df.columns
    assert df.iloc[0]["Source"] == "test.csv"
    # Check if reindexing worked (Reason column should exist even if not in CSV)
    assert "Reason" in df.columns