# storage_service.py - Simplified S3-Only Storage Service

"""
Simplified S3-only storage service with true metadata-only access support.
Provides clean interface: get, save, delete, list, find methods.
"""

import logging
import os
import pickle
import gzip
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
load_dotenv()

# S3 imports - required for this service
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    raise ImportError("boto3 is required for storage service. Install with: pip install boto3")

logger = logging.getLogger(__name__)


class S3StorageService:
    """Simple S3 storage service with metadata-only access optimization"""
    
    def __init__(self, bucket_name: str, prefix: str = "", region: str = None):
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
            
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ""
        self.region = region
        
        logger.info(f"S3Storage initializing: bucket={bucket_name}, prefix={self.prefix}, region={region}")
        
        # Initialize S3 client
        try:
            # Check for local development environment
            use_local_s3 = os.getenv('USE_LOCAL_S3', 'false').lower() == 'true'
            
            if use_local_s3:
                # Configure for local S3 (LocalStack or MinIO)
                local_endpoint = os.getenv('LOCAL_S3_ENDPOINT', 'http://localhost:4566')
                logger.info(f"Using local S3 endpoint: {local_endpoint}")
                
                # Create session with explicit credentials for LocalStack
                session = boto3.Session(
                    aws_access_key_id='test',
                    aws_secret_access_key='test',
                    region_name=region or 'us-east-1'
                )
                
                self.s3_client = session.client(
                    's3',
                    endpoint_url=local_endpoint,
                    use_ssl=False,
                    verify=False
                )
            else:
                # Production AWS S3 - validate region first
                effective_region = region or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
                
                # Validate region format
                if not effective_region or ':' in effective_region:
                    logger.warning(f"Invalid region format detected: '{effective_region}'. Using us-east-1 as fallback.")
                    effective_region = 'us-east-1'
                
                self.s3_client = boto3.client('s3', region_name=effective_region)
                
            # Test connection
            self._test_connection()
            logger.info(f"S3Storage initialized successfully")
            
        except NoCredentialsError:
            raise RuntimeError("AWS credentials not found. Ensure IAM role is properly configured.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
    
    def _test_connection(self):
        """Test S3 connection and bucket access"""
        use_local_s3 = os.getenv('USE_LOCAL_S3', 'false').lower() == 'true'
        
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket '{self.bucket_name}' exists and is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # For local development, try to create the bucket
                if use_local_s3:
                    try:
                        logger.info(f"Bucket '{self.bucket_name}' not found. Creating local S3 bucket...")
                        
                        # Create bucket with proper configuration for LocalStack
                        create_bucket_config = {}
                        if self.region and self.region != 'us-east-1':
                            create_bucket_config['CreateBucketConfiguration'] = {
                                'LocationConstraint': self.region
                            }
                        
                        if create_bucket_config:
                            self.s3_client.create_bucket(
                                Bucket=self.bucket_name, 
                                **create_bucket_config
                            )
                        else:
                            self.s3_client.create_bucket(Bucket=self.bucket_name)
                        
                        logger.info(f"Local S3 bucket '{self.bucket_name}' created successfully")
                        
                        # Verify bucket was created
                        self.s3_client.head_bucket(Bucket=self.bucket_name)
                        logger.info(f"Verified bucket '{self.bucket_name}' is accessible")
                        return
                        
                    except Exception as create_error:
                        logger.error(f"Failed to create local S3 bucket '{self.bucket_name}': {create_error}")
                        logger.error(f"Error type: {type(create_error).__name__}")
                        raise RuntimeError(f"Failed to create local S3 bucket: {create_error}")
                else:
                    raise RuntimeError(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                raise RuntimeError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                logger.error(f"S3 bucket access error: {error_code} - {e}")
                raise RuntimeError(f"S3 bucket access error: {e}")
        except Exception as e:
            # Handle connection errors for local development
            if use_local_s3:
                local_endpoint = os.getenv('LOCAL_S3_ENDPOINT', 'http://localhost:4566')
                logger.error(f"Cannot connect to local S3 endpoint '{local_endpoint}': {e}")
                logger.info("Make sure LocalStack is running with: docker run --rm -d -p 4566:4566 localstack/localstack")
                raise RuntimeError(f"Cannot connect to local S3 endpoint '{local_endpoint}'. "
                                 f"Make sure LocalStack is running: docker run --rm -d -p 4566:4566 localstack/localstack")
            else:
                raise RuntimeError(f"S3 connection failed: {e}")
    
    def _get_s3_key(self, key: str) -> str:
        """Convert storage key to S3 object key"""
        return f"{self.prefix}{key}.pkl"
    
    def _get_metadata_key(self, key: str) -> str:
        """Get S3 key for metadata-only object"""
        return f"{self.prefix}{key}.metadata.json"
    
    def save(self, key: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Save data with optional metadata for efficient retrieval
        
        Args:
            key: Storage key
            data: Data to store
            metadata: Optional metadata for efficient access (will be stored separately)
            
        Returns:
            bool: Success status
        """
        import time
        start_time = time.time()
        
        try:
            # Serialize main data
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if large
            if len(serialized_data) > 1024 * 1024:  # 1MB threshold
                compressed_data = gzip.compress(serialized_data, compresslevel=1)
                is_compressed = True
            else:
                compressed_data = serialized_data
                is_compressed = False
            
            # Store main data
            s3_key = self._get_s3_key(key)
            extra_args = {
                'Metadata': {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'compressed': str(is_compressed),
                    'original_size': str(len(serialized_data))
                }
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=compressed_data,
                **extra_args
            )
            
            # Store metadata separately for efficient access
            if metadata:
                metadata_key = self._get_metadata_key(key)
                metadata_with_timestamp = {
                    **metadata,
                    '_storage_metadata': {
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'data_key': s3_key,
                        'compressed': is_compressed,
                        'original_size': len(serialized_data)
                    }
                }
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(metadata_with_timestamp, default=str),
                    ContentType='application/json'
                )
            
            elapsed = time.time() - start_time
            logger.info(f"S3Storage SAVE: {key} saved in {elapsed:.3f}s ({len(compressed_data)} bytes)")
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage SAVE FAILED: {key} in {elapsed:.3f}s - {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get full data object
        
        Args:
            key: Storage key
            
        Returns:
            Stored data or None if not found
        """
        import time
        start_time = time.time()
        
        try:
            s3_key = self._get_s3_key(key)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            raw_data = response['Body'].read()
            
            # Check if compressed
            metadata = response.get('Metadata', {})
            is_compressed = metadata.get('compressed', 'False').lower() == 'true'
            
            if is_compressed:
                decompressed_data = gzip.decompress(raw_data)
                data = pickle.loads(decompressed_data)
            else:
                data = pickle.loads(raw_data)
            
            elapsed = time.time() - start_time
            logger.info(f"S3Storage GET: {key} retrieved in {elapsed:.3f}s")
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.debug(f"S3Storage GET: {key} not found")
                return None
            logger.error(f"S3Storage GET FAILED: {key} - {e}")
            return None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage GET ERROR: {key} in {elapsed:.3f}s - {e}")
            return None
    
    def get_metadata_only(self, key: str) -> Optional[Dict]:
        """Get only metadata without loading full data - TRUE metadata-only access
        
        Args:
            key: Storage key
            
        Returns:
            Metadata dict or None if not found
        """
        import time
        start_time = time.time()
        
        try:
            metadata_key = self._get_metadata_key(key)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
            metadata_json = response['Body'].read().decode('utf-8')
            metadata = json.loads(metadata_json)
            
            elapsed = time.time() - start_time
            logger.debug(f"S3Storage METADATA: {key} retrieved in {elapsed:.3f}s")
            return metadata
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.debug(f"S3Storage METADATA: {key} metadata not found")
                return None
            logger.error(f"S3Storage METADATA FAILED: {key} - {e}")
            return None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage METADATA ERROR: {key} in {elapsed:.3f}s - {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete data and its metadata
        
        Args:
            key: Storage key
            
        Returns:
            bool: Success status
        """
        try:
            # Delete main data
            s3_key = self._get_s3_key(key)
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            
            # Delete metadata if it exists
            try:
                metadata_key = self._get_metadata_key(key)
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=metadata_key)
            except:
                pass  # Metadata might not exist
            
            logger.debug(f"S3Storage DELETE: {key} deleted")
            return True
            
        except Exception as e:
            logger.error(f"S3Storage DELETE FAILED: {key} - {e}")
            return False
    
    def list(self, prefix: str = "") -> List[str]:
        """List all storage keys with optional prefix filter
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of storage keys
        """
        try:
            search_prefix = f"{self.prefix}{prefix}" if prefix else self.prefix
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=search_prefix)
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        # Only include main data files, not metadata files
                        if s3_key.startswith(self.prefix) and s3_key.endswith('.pkl'):
                            storage_key = s3_key[len(self.prefix):-4]  # Remove prefix and .pkl
                            keys.append(storage_key)
            
            logger.debug(f"S3Storage LIST: found {len(keys)} keys with prefix '{prefix}'")
            return keys
            
        except Exception as e:
            logger.error(f"S3Storage LIST FAILED: prefix '{prefix}' - {e}")
            return []
    
    def find(self, criteria: Dict[str, Any]) -> List[str]:
        """Find storage keys based on metadata criteria
        
        Args:
            criteria: Dictionary of metadata criteria to match
            
        Returns:
            List of matching storage keys
        """
        try:
            # Get all keys
            all_keys = self.list()
            matching_keys = []
            
            for key in all_keys:
                metadata = self.get_metadata_only(key)
                if metadata:
                    # Check if metadata matches all criteria
                    matches = True
                    for criterion_key, criterion_value in criteria.items():
                        if metadata.get(criterion_key) != criterion_value:
                            matches = False
                            break
                    
                    if matches:
                        matching_keys.append(key)
            
            logger.debug(f"S3Storage FIND: found {len(matching_keys)} keys matching criteria")
            return matching_keys
            
        except Exception as e:
            logger.error(f"S3Storage FIND FAILED: {e}")
            return []
    
    def exists(self, key: str) -> bool:
        """Check if key exists
        
        Args:
            key: Storage key
            
        Returns:
            bool: True if exists
        """
        try:
            s3_key = self._get_s3_key(key)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"S3Storage EXISTS FAILED: {key} - {e}")
            return False
        except Exception as e:
            logger.error(f"S3Storage EXISTS ERROR: {key} - {e}")
            return False


def create_storage_service() -> S3StorageService:
    """Factory function to create S3 storage service based on configuration"""
    
    bucket_name = os.getenv('S3_BUCKET_NAME','test')
    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME environment variable is required. Please set S3_BUCKET_NAME.")
    
    s3_prefix = os.getenv('S3_PREFIX', 'prism-ai-storage')
    aws_region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION'))
    
    logger.info(f"Creating S3 storage service - bucket: {bucket_name}, prefix: {s3_prefix}, region: {aws_region}")
    
    return S3StorageService(
        bucket_name=bucket_name,
        prefix=s3_prefix,
        region=aws_region
    )


# Initialize storage services for different data types
logger.info("Initializing storage services...")

# Create separate prefixes for different data types for better organization
_main_storage = create_storage_service()

# Create logical containers with different prefixes
class StorageContainer:
    """Wrapper to provide prefix-based separation"""
    
    def __init__(self, storage_service: S3StorageService, container_prefix: str):
        self.storage = storage_service
        self.container_prefix = container_prefix
    
    def save(self, key: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        return self.storage.save(f"{self.container_prefix}/{key}", data, metadata)
    
    def get(self, key: str) -> Optional[Any]:
        return self.storage.get(f"{self.container_prefix}/{key}")
    
    def get_metadata_only(self, key: str) -> Optional[Dict]:
        return self.storage.get_metadata_only(f"{self.container_prefix}/{key}")
    
    def delete(self, key: str) -> bool:
        return self.storage.delete(f"{self.container_prefix}/{key}")
    
    def list(self, prefix: str = "") -> List[str]:
        full_prefix = f"{self.container_prefix}/{prefix}" if prefix else f"{self.container_prefix}/"
        keys = self.storage.list(full_prefix)
        # Remove container prefix from returned keys
        container_prefix_with_slash = f"{self.container_prefix}/"
        return [key[len(container_prefix_with_slash):] for key in keys if key.startswith(container_prefix_with_slash)]
    
    def find(self, criteria: Dict[str, Any]) -> List[str]:
        # Find within this container's prefix
        all_keys = self.list()
        matching_keys = []
        
        for key in all_keys:
            metadata = self.get_metadata_only(key)
            if metadata:
                matches = True
                for criterion_key, criterion_value in criteria.items():
                    if metadata.get(criterion_key) != criterion_value:
                        matches = False
                        break
                
                if matches:
                    matching_keys.append(key)
        
        return matching_keys
    
    def exists(self, key: str) -> bool:
        return self.storage.exists(f"{self.container_prefix}/{key}")
    
    def keys_only(self) -> List[str]:
        """Backward compatibility method - use list() instead"""
        return self.list()


# Create storage containers for different data types
uploaded_files = StorageContainer(_main_storage, "uploaded_files")
extractions = StorageContainer(_main_storage, "extractions")
comparisons = StorageContainer(_main_storage, "comparisons")
reconciliations = StorageContainer(_main_storage, "reconciliations")

logger.info("Storage services initialized successfully")


# Utility functions
def get_storage_info() -> Dict[str, Any]:
    """Get information about current storage configuration"""
    return {
        'storage_type': 's3',
        'bucket_name': _main_storage.bucket_name,
        'prefix': _main_storage.prefix,
        'region': _main_storage.region,
        's3_available': S3_AVAILABLE
    }


# Legacy compatibility (if needed)
def get_storage_backend():
    """Get the main storage backend for direct access"""
    return _main_storage