import pandera.pandas as pa
from pandera.typing import DataFrame
from typing import Optional

class KeywordSchema(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)
    required_kw: Optional[str]
    headers: Optional[str]


class KeywordSchemaRaw(pa.DataFrameModel):
    brand: str
    product: str
    keyword: str
    required_product: str = pa.Field(nullable=True)


class ChatSchema(pa.DataFrameModel):
    Source: str
    Date1: str = pa.Field(nullable=True)
    Date2: str
    Time: str
    userPhone: str = pa.Field(coerce=True)
    quotedMessage: str = pa.Field(nullable=True)
    messageBody: str = pa.Field(nullable=True)
    mediaType: str = pa.Field(nullable=True)
    mediaCaption: str = pa.Field(nullable=True)
    Reason: str = pa.Field(nullable=True)