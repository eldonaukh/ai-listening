import pandera.pandas as pa
from typing import NamedTuple

class KeywordSchemaRaw(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)


class KeywordSchema(KeywordSchemaRaw):
    required_kw: str = pa.Field(nullable=True)
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
    required_kw: str | None