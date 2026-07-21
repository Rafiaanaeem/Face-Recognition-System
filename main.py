from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from app.api.routes import router 

app = FastAPI(title="Face Recognition API")
app.openapi_version = "3.0.3"

app.include_router(router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version="3.0.3",
        routes=app.routes,
    )

    def fix_file_schemas(schema_dict):
        if isinstance(schema_dict, dict):
            if schema_dict.get("contentMediaType") == "application/octet-stream":
                schema_dict.pop("contentMediaType", None)
                schema_dict["format"] = "binary"
            for value in schema_dict.values():
                fix_file_schemas(value)
        elif isinstance(schema_dict, list):
            for item in schema_dict:
                fix_file_schemas(item)

    fix_file_schemas(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi