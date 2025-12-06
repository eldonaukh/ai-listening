import pandera.pandas as pa


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
    Reason: str = pa.Field(nullable=True)
