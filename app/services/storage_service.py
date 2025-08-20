# storage.py - Enhanced Storage Module with S3 Integration

"""
Enhanced storage module supporting both local and S3 storage backends.
Provides centralized storage that can be imported by other modules with seamless switching.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List

# Optional S3 imports - graceful fallback if boto3 not available
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass


class LocalStorageBackend(StorageBackend):
    """Local dictionary-based storage backend"""

    def __init__(self):
        self.storage: Dict[str, Any] = {}
        logger.info("LocalStorage initialized: Using in-memory dictionary storage")

    def get(self, key: str) -> Optional[Any]:
        logger.debug(f"LocalStorage GET: {key}")
        return self.storage.get(key)

    def set(self, key: str, value: Any) -> bool:
        try:
            self.storage[key] = value
            logger.debug(f"LocalStorage SET: {key} (size: {len(str(value)) if value else 0} chars)")
            return True
        except Exception as e:
            logger.error(f"LocalStorage SET failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            if key in self.storage:
                del self.storage[key]
                logger.debug(f"LocalStorage DELETE: {key}")
                return True
            logger.debug(f"LocalStorage DELETE: {key} not found")
            return False
        except Exception as e:
            logger.error(f"LocalStorage DELETE failed for key {key}: {e}")
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        if not prefix:
            keys = list(self.storage.keys())
            logger.debug(f"LocalStorage LIST_KEYS: returning {len(keys)} keys")
            return keys
        keys = [key for key in self.storage.keys() if key.startswith(prefix)]
        logger.debug(f"LocalStorage LIST_KEYS: prefix '{prefix}' returned {len(keys)} keys")
        return keys

    def exists(self, key: str) -> bool:
        exists = key in self.storage
        logger.debug(f"LocalStorage EXISTS: {key} -> {exists}")
        return exists

    def items(self):
        """For backward compatibility with dict-like access"""
        return self.storage.items()

    def keys(self):
        """For backward compatibility with dict-like access"""
        return self.storage.keys()

    def values(self):
        """For backward compatibility with dict-like access"""
        return self.storage.values()


class S3StorageBackend(StorageBackend):
    """S3-based storage backend"""

    def __init__(self, bucket_name: str, prefix: str = "", region: str = None):
        if not S3_AVAILABLE:
            logger.error("S3Storage initialization failed: boto3 not available")
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")

        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ""
        self.region = region

        logger.info(f"S3Storage initializing: bucket={bucket_name}, prefix={self.prefix}, region={region}")

        # Initialize S3 client - will use IAM role credentials automatically
        try:
            import time
            start_time = time.time()
            
            if region:
                self.s3_client = boto3.client('s3', region_name=region)
                logger.debug(f"S3 client created with region: {region}")
            else:
                self.s3_client = boto3.client('s3')
                logger.debug("S3 client created with default region")

            # Test connection and bucket access
            self._test_connection()
            
            init_time = time.time() - start_time
            logger.info(f"S3Storage initialized successfully in {init_time:.3f}s: bucket={bucket_name}, prefix={self.prefix}")

        except NoCredentialsError:
            logger.error("S3Storage initialization failed: AWS credentials not found")
            raise RuntimeError("AWS credentials not found. Ensure IAM role is properly configured.")
        except Exception as e:
            logger.error(f"S3Storage initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize S3 client: {e}")

    def _test_connection(self):
        """Test S3 connection and bucket access"""
        try:
            logger.debug(f"S3Storage testing connection to bucket: {self.bucket_name}")
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"S3Storage connection test successful for bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3Storage connection test failed: Bucket '{self.bucket_name}' not found")
                raise RuntimeError(f"S3 bucket '{self.bucket_name}' not found")
            elif error_code == '403':
                logger.error(f"S3Storage connection test failed: Access denied to bucket '{self.bucket_name}'")
                raise RuntimeError(f"Access denied to S3 bucket '{self.bucket_name}'")
            else:
                logger.error(f"S3Storage connection test failed: {e}")
                raise RuntimeError(f"S3 bucket access error: {e}")

    def _get_s3_key(self, key: str) -> str:
        """Convert storage key to S3 object key"""
        return f"{self.prefix}{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        import time
        start_time = time.time()
        try:
            s3_key = self._get_s3_key(key)
            logger.debug(f"S3Storage GET: {key} -> {s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = pickle.loads(response['Body'].read())
            
            elapsed = time.time() - start_time
            logger.debug(f"S3Storage GET success: {key} in {elapsed:.3f}s")
            return data
            
        except ClientError as e:
            elapsed = time.time() - start_time
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.debug(f"S3Storage GET: {key} not found in {elapsed:.3f}s")
                return None
            logger.error(f"S3Storage GET failed: {key} in {elapsed:.3f}s - {e}")
            return None
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage GET deserialization failed: {key} in {elapsed:.3f}s - {e}")
            return None

    def set(self, key: str, value: Any) -> bool:
        import time
        start_time = time.time()
        try:
            s3_key = self._get_s3_key(key)
            serialized_data = pickle.dumps(value)
            data_size = len(serialized_data)
            
            logger.debug(f"S3Storage SET: {key} -> {s3_key} (size: {data_size} bytes)")

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=serialized_data,
                Metadata={
                    'created_at': datetime.utcnow().isoformat(),
                    'storage_type': 'enhanced_storage'
                }
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"S3Storage SET success: {key} in {elapsed:.3f}s ({data_size} bytes)")
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage SET failed: {key} in {elapsed:.3f}s - {e}")
            return False

    def delete(self, key: str) -> bool:
        import time
        start_time = time.time()
        try:
            s3_key = self._get_s3_key(key)
            logger.debug(f"S3Storage DELETE: {key} -> {s3_key}")
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            
            elapsed = time.time() - start_time
            logger.debug(f"S3Storage DELETE success: {key} in {elapsed:.3f}s")
            return True
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage DELETE failed: {key} in {elapsed:.3f}s - {e}")
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        import time
        start_time = time.time()
        try:
            search_prefix = f"{self.prefix}{prefix}" if prefix else self.prefix
            logger.debug(f"S3Storage LIST_KEYS: searching with prefix '{search_prefix}'")

            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=search_prefix)

            keys = []
            page_count = 0
            for page in pages:
                page_count += 1
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Convert S3 key back to storage key
                        s3_key = obj['Key']
                        if s3_key.startswith(self.prefix) and s3_key.endswith('.pkl'):
                            storage_key = s3_key[len(self.prefix):-4]  # Remove prefix and .pkl
                            keys.append(storage_key)

            elapsed = time.time() - start_time
            logger.debug(f"S3Storage LIST_KEYS success: prefix '{prefix}' returned {len(keys)} keys from {page_count} pages in {elapsed:.3f}s")
            return keys
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage LIST_KEYS failed: prefix '{prefix}' in {elapsed:.3f}s - {e}")
            return []

    def exists(self, key: str) -> bool:
        import time
        start_time = time.time()
        try:
            s3_key = self._get_s3_key(key)
            logger.debug(f"S3Storage EXISTS: {key} -> {s3_key}")
            
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            
            elapsed = time.time() - start_time
            logger.debug(f"S3Storage EXISTS: {key} -> True in {elapsed:.3f}s")
            return True
            
        except ClientError as e:
            elapsed = time.time() - start_time
            if e.response['Error']['Code'] == '404':
                logger.debug(f"S3Storage EXISTS: {key} -> False in {elapsed:.3f}s")
                return False
            logger.error(f"S3Storage EXISTS failed: {key} in {elapsed:.3f}s - {e}")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"S3Storage EXISTS failed: {key} in {elapsed:.3f}s - {e}")
            return False

    def items(self):
        """For backward compatibility - WARNING: This loads all data into memory"""
        logger.warning("items() loads all S3 data into memory. Consider using list_keys() for large datasets.")
        for key in self.list_keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def keys(self):
        """For backward compatibility"""
        return self.list_keys()

    def values(self):
        """For backward compatibility - WARNING: This loads all data into memory"""
        logger.warning("values() loads all S3 data into memory. Consider iterating through keys.")
        for key in self.list_keys():
            value = self.get(key)
            if value is not None:
                yield value


class EnhancedStorage:
    """Enhanced storage class that provides dict-like interface with pluggable backends"""

    def __init__(self, backend: StorageBackend):
        self.backend = backend

    def __getitem__(self, key: str):
        value = self.backend.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any):
        success = self.backend.set(key, value)
        if not success:
            raise RuntimeError(f"Failed to set key: {key}")

    def __delitem__(self, key: str):
        success = self.backend.delete(key)
        if not success:
            raise KeyError(key)

    def __contains__(self, key: str):
        return self.backend.exists(key)

    def __len__(self):
        return len(self.backend.list_keys())

    def get(self, key: str, default=None):
        value = self.backend.get(key)
        return value if value is not None else default

    def items(self):
        return self.backend.items()

    def keys(self):
        return self.backend.keys()

    def values(self):
        return self.backend.values()

    def pop(self, key: str, default=None):
        value = self.get(key, default)
        if key in self:
            del self[key]
        return value

    def clear(self):
        """Clear all items - use with caution"""
        for key in list(self.keys()):
            del self[key]


def create_storage_backend() -> StorageBackend:
    """Factory function to create appropriate storage backend based on configuration"""

    storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
    logger.info(f"StorageService: Initializing storage backend type: {storage_type}")

    if storage_type == 's3':
        if not S3_AVAILABLE:
            logger.error("StorageService: boto3 not available for S3 storage")
            raise RuntimeError("S3 storage requested but boto3 not installed. Install with: pip install boto3")

        bucket_name = os.getenv('S3_BUCKET_NAME')
        if not bucket_name:
            logger.error("StorageService: S3_BUCKET_NAME environment variable is required when STORAGE_TYPE=s3")
            raise RuntimeError("S3 storage requested but S3_BUCKET_NAME environment variable not set. Please set S3_BUCKET_NAME.")

        try:
            s3_prefix = os.getenv('S3_PREFIX', 'financial-data-storage')
            aws_region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION'))
            
            logger.info(f"StorageService: Attempting S3 initialization - bucket: {bucket_name}, prefix: {s3_prefix}, region: {aws_region}")
            
            return S3StorageBackend(
                bucket_name=bucket_name,
                prefix=s3_prefix,
                region=aws_region
            )
        except Exception as e:
            logger.error(f"StorageService: Failed to initialize S3 storage backend: {e}")
            raise RuntimeError(f"S3 storage initialization failed: {e}. Check AWS credentials, bucket permissions, and network connectivity.")

    else:
        logger.info(f"StorageService: Using local storage backend (STORAGE_TYPE={storage_type})")
        return LocalStorageBackend()


# Initialize storage backends
logger.info("StorageService: Creating storage backend...")
_backend = create_storage_backend()

# Create enhanced storage instances with dict-like interface
logger.info("StorageService: Initializing storage containers...")
uploaded_files = EnhancedStorage(_backend)
extractions = EnhancedStorage(_backend)
comparisons = EnhancedStorage(_backend)
reconciliations = EnhancedStorage(_backend)

logger.info(f"StorageService: Successfully initialized {_backend.__class__.__name__} with 4 storage containers")


# Utility functions for storage management
def get_storage_info() -> Dict[str, Any]:
    """Get information about current storage configuration"""
    storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
    info = {
        'storage_type': storage_type,
        'backend_class': _backend.__class__.__name__,
        's3_available': S3_AVAILABLE
    }

    if isinstance(_backend, S3StorageBackend):
        info.update({
            'bucket_name': _backend.bucket_name,
            'prefix': _backend.prefix,
            'region': _backend.region
        })

    return info


def switch_storage_backend(backend: StorageBackend):
    """Switch to a different storage backend - use with caution"""
    global _backend, uploaded_files, extractions, comparisons, reconciliations

    logger.warning("Switching storage backend - existing data may become inaccessible")
    _backend = backend

    # Recreate storage instances with new backend
    uploaded_files = EnhancedStorage(_backend)
    extractions = EnhancedStorage(_backend)
    comparisons = EnhancedStorage(_backend)
    reconciliations = EnhancedStorage(_backend)


# Initialize logging
logger.info(f"Storage initialized: {get_storage_info()}")
