"""
Dependency injection utility for the application.
Provides a simple service locator pattern for managing dependencies.
"""

from typing import Dict, Any, Type, TypeVar, Optional, Callable

T = TypeVar('T')

class ServiceRegistry:
    """
    A simple service registry for dependency injection.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        
    def register(self, service_type: Type[T], instance: T, name: Optional[str] = None) -> None:
        """
        Register a service instance.
        
        Args:
            service_type: The type of the service
            instance: The service instance
            name: Optional name for the service (for multiple instances of the same type)
        """
        key = self._get_key(service_type, name)
        self._services[key] = instance
        
    def register_factory(self, service_type: Type[T], factory: Callable[[], T], name: Optional[str] = None) -> None:
        """
        Register a factory function for lazy initialization of a service.
        
        Args:
            service_type: The type of the service
            factory: A function that creates the service instance
            name: Optional name for the service (for multiple instances of the same type)
        """
        key = self._get_key(service_type, name)
        self._factories[key] = factory
        
    def get(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """
        Get a service instance.
        
        Args:
            service_type: The type of the service
            name: Optional name for the service (for multiple instances of the same type)
            
        Returns:
            The service instance
            
        Raises:
            KeyError: If the service is not registered
        """
        key = self._get_key(service_type, name)
        
        # Check if we have an instance
        if key in self._services:
            return self._services[key]
            
        # Check if we have a factory
        if key in self._factories:
            # Create the instance and cache it
            instance = self._factories[key]()
            self._services[key] = instance
            return instance
            
        raise KeyError(f"Service not registered: {key}")
        
    def has(self, service_type: Type[T], name: Optional[str] = None) -> bool:
        """
        Check if a service is registered.
        
        Args:
            service_type: The type of the service
            name: Optional name for the service (for multiple instances of the same type)
            
        Returns:
            True if the service is registered, False otherwise
        """
        key = self._get_key(service_type, name)
        return key in self._services or key in self._factories
        
    def _get_key(self, service_type: Type, name: Optional[str] = None) -> str:
        """
        Get the key for a service.
        
        Args:
            service_type: The type of the service
            name: Optional name for the service
            
        Returns:
            The key for the service
        """
        type_name = service_type.__name__
        return f"{type_name}:{name}" if name else type_name


# Global service registry instance
services = ServiceRegistry()


def get_service(service_type: Type[T], name: Optional[str] = None) -> T:
    """
    Get a service instance from the global registry.
    
    Args:
        service_type: The type of the service
        name: Optional name for the service
        
    Returns:
        The service instance
    """
    return services.get(service_type, name)


def register_service(service_type: Type[T], instance: T, name: Optional[str] = None) -> None:
    """
    Register a service instance in the global registry.
    
    Args:
        service_type: The type of the service
        instance: The service instance
        name: Optional name for the service
    """
    services.register(service_type, instance, name)


def register_factory(service_type: Type[T], factory: Callable[[], T], name: Optional[str] = None) -> None:
    """
    Register a factory function in the global registry.
    
    Args:
        service_type: The type of the service
        factory: A function that creates the service instance
        name: Optional name for the service
    """
    services.register_factory(service_type, factory, name)
