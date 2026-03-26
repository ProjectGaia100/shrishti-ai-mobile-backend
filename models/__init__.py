"""
Data Models
Simple data classes and models for the application
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class Location:
    """Location data model"""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        """Validate coordinates"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude
        }

@dataclass
class ChatMessage:
    """Chat message data model"""
    message: str
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set default timestamp"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class ChatResponse:
    """Chat response data model"""
    response: str
    status: str
    model: Optional[str] = None
    attempt: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set default timestamp"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'response': self.response,
            'status': self.status,
            'model': self.model,
            'attempt': self.attempt,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class SatelliteRequest:
    """Satellite data request model"""
    location: Location
    start_date: str
    end_date: str
    collection: str = 'COPERNICUS/S2_SR'
    cloud_filter: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'location': self.location.to_dict(),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'collection': self.collection,
            'cloud_filter': self.cloud_filter
        }

@dataclass
class RegionRequest:
    """Region data request model"""
    bounds: List[List[float]]
    start_date: str
    end_date: str
    scale: int = 10
    
    def __post_init__(self):
        """Validate bounds"""
        if len(self.bounds) < 3:
            raise ValueError("Bounds must contain at least 3 coordinate pairs")
        
        for i, coord in enumerate(self.bounds):
            if len(coord) != 2:
                raise ValueError(f"Invalid coordinate at index {i}")
            
            lon, lat = coord
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise ValueError(f"Invalid coordinates at index {i}: [{lon}, {lat}]")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'bounds': self.bounds,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'scale': self.scale
        }

@dataclass
class ServiceStatus:
    """Service status model"""
    service_name: str
    status: str
    initialized: bool
    timestamp: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default timestamp"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'service_name': self.service_name,
            'status': self.status,
            'initialized': self.initialized,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'details': self.details
        }

@dataclass
class ErrorResponse:
    """Error response model"""
    error: str
    status: str = 'error'
    timestamp: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default timestamp"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error': self.error,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'details': self.details
        }