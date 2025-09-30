from fastapi import FastAPI
from fastapi.routing import APIRoute
from server.main import app

def list_routes(app: FastAPI):
    """List all registered routes in the FastAPI application."""
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append({
                "path": route.path,
                "name": route.name,
                "methods": list(route.methods) if hasattr(route, 'methods') else [],
                "endpoint": route.endpoint.__name__ if hasattr(route.endpoint, '__name__') else str(route.endpoint)
            })
    return routes

if __name__ == "__main__":
    routes = list_routes(app)
    for route in routes:
        print(f"{', '.join(route['methods'])} {route['path']} -> {route['endpoint']}")
    print(f"\nTotal routes: {len(routes)}")
