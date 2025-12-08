import pandera.pandas as pa
from typing import NamedTuple, Literal
from pydantic import BaseModel

class KeywordSchemaRaw(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)


class KeywordSchema(KeywordSchemaRaw):
    required_keyword: str = pa.Field(nullable=True)
    headers: str


class ChatSchemaRaw(pa.DataFrameModel):    
    Date1: str = pa.Field(nullable=True)
    Date2: str
    Time: str
    userPhone: str = pa.Field(coerce=True)
    quotedMessage: str = pa.Field(nullable=True)
    messageBody: str = pa.Field(nullable=True)
    mediaType: str = pa.Field(nullable=True)
    mediaCaption: str = pa.Field(nullable=True)


class ChatSchema(ChatSchemaRaw):
    Source: str
    Group: str = pa.Field(nullable=True)
    Reason: str = pa.Field(nullable=True)


class KeywordRow(NamedTuple):
    brand: str
    product: str
    keyword: str
    required_product: str | None
    required_keyword: str | None

class ChatRow(NamedTuple):
    Index: str
    Source: str
    Group: str | None
    Date1: str
    Date2: str
    Time: str
    userPhone: str
    quotedMessage: str
    messageBody: str
    mediaType: str
    mediaCaption: str
    Reason: str

class SentimentResponse(BaseModel):
    sentiment: Literal["P", "N", "I"]
    reason: str