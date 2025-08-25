# Delta Storage Service - S3-based storage for delta results similar to reconciliation service
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


class OptimizedDeltaStorage:
    """Optimized storage for delta results using S3-based storage service"""
    
    def __init__(self):
        # Use uploaded_files storage so viewer routes can access the results
        from app.services.storage_service import uploaded_files
        self.storage = uploaded_files
        logger.info(f"DeltaStorage: Initialized with uploaded_files storage for viewer compatibility")

    def store_results(self, delta_id: str, results: Dict[str, pd.DataFrame]) -> bool:
        """Store delta results as separate files for frontend compatibility"""
        import time
        start_time = time.time()
        
        try:
            # Calculate data size for logging
            unchanged_count = len(results['unchanged'])
            amended_count = len(results['amended'])
            deleted_count = len(results['deleted'])
            newly_added_count = len(results['newly_added'])
            total_records = unchanged_count + amended_count + deleted_count + newly_added_count
            
            logger.info(f"DeltaStorage STORE: {delta_id} - {total_records} total records "
                       f"({unchanged_count} unchanged, {amended_count} amended, "
                       f"{deleted_count} deleted, {newly_added_count} newly_added)")
            
            # Create timestamp for all files as ISO string
            timestamp_iso = datetime.utcnow().isoformat()
            
            # Store each result type as a separate file for frontend compatibility
            all_success = True
            
            # 1. Store unchanged results
            if unchanged_count > 0:
                unchanged_columns = list(results['unchanged'].columns)
                unchanged_info = {
                    'file_id': f"{delta_id}_unchanged",
                    'filename': f'delta_{delta_id}_unchanged.xlsx',
                    'custom_name': None,
                    'file_type': 'delta',
                    'file_source': 'delta',
                    'total_rows': unchanged_count,
                    'total_columns': len(unchanged_columns),
                    'columns': unchanged_columns,
                    'upload_time': timestamp_iso,
                    'file_size_mb': 0,  # We don't calculate file size for delta results
                    'data_types': {col: str(dtype) for col, dtype in results['unchanged'].dtypes.items()},
                    'delta_id': delta_id,
                    'result_type': 'unchanged'
                }
                
                unchanged_data = {
                    'info': unchanged_info,
                    'data': results['unchanged']
                }
                
                success = self.storage.save(f"{delta_id}_unchanged", unchanged_data, unchanged_info)
                all_success = all_success and success
                if success:
                    logger.info(f"DeltaStorage: Stored unchanged results ({unchanged_count} records)")
            
            # 2. Store amended results
            if amended_count > 0:
                amended_columns = list(results['amended'].columns)
                amended_info = {
                    'file_id': f"{delta_id}_amended",
                    'filename': f'delta_{delta_id}_amended.xlsx',
                    'custom_name': None,
                    'file_type': 'delta',
                    'file_source': 'delta',
                    'total_rows': amended_count,
                    'total_columns': len(amended_columns),
                    'columns': amended_columns,
                    'upload_time': timestamp_iso,
                    'file_size_mb': 0,
                    'data_types': {col: str(dtype) for col, dtype in results['amended'].dtypes.items()},
                    'delta_id': delta_id,
                    'result_type': 'amended'
                }
                
                amended_data = {
                    'info': amended_info,
                    'data': results['amended']
                }
                
                success = self.storage.save(f"{delta_id}_amended", amended_data, amended_info)
                all_success = all_success and success
                if success:
                    logger.info(f"DeltaStorage: Stored amended results ({amended_count} records)")
            
            # 3. Store deleted results
            if deleted_count > 0:
                deleted_columns = list(results['deleted'].columns)
                deleted_info = {
                    'file_id': f"{delta_id}_deleted",
                    'filename': f'delta_{delta_id}_deleted.xlsx',
                    'custom_name': None,
                    'file_type': 'delta',
                    'file_source': 'delta',
                    'total_rows': deleted_count,
                    'total_columns': len(deleted_columns),
                    'columns': deleted_columns,
                    'upload_time': timestamp_iso,
                    'file_size_mb': 0,
                    'data_types': {col: str(dtype) for col, dtype in results['deleted'].dtypes.items()},
                    'delta_id': delta_id,
                    'result_type': 'deleted'
                }
                
                deleted_data = {
                    'info': deleted_info,
                    'data': results['deleted']
                }
                
                success = self.storage.save(f"{delta_id}_deleted", deleted_data, deleted_info)
                all_success = all_success and success
                if success:
                    logger.info(f"DeltaStorage: Stored deleted results ({deleted_count} records)")
            
            # 4. Store newly_added results
            if newly_added_count > 0:
                newly_added_columns = list(results['newly_added'].columns)
                newly_added_info = {
                    'file_id': f"{delta_id}_newly_added",
                    'filename': f'delta_{delta_id}_newly_added.xlsx',
                    'custom_name': None,
                    'file_type': 'delta',
                    'file_source': 'delta',
                    'total_rows': newly_added_count,
                    'total_columns': len(newly_added_columns),
                    'columns': newly_added_columns,
                    'upload_time': timestamp_iso,
                    'file_size_mb': 0,
                    'data_types': {col: str(dtype) for col, dtype in results['newly_added'].dtypes.items()},
                    'delta_id': delta_id,
                    'result_type': 'newly_added'
                }
                
                newly_added_data = {
                    'info': newly_added_info,
                    'data': results['newly_added']
                }
                
                success = self.storage.save(f"{delta_id}_newly_added", newly_added_data, newly_added_info)
                all_success = all_success and success
                if success:
                    logger.info(f"DeltaStorage: Stored newly_added results ({newly_added_count} records)")

            # Store metadata file with summary information
            metadata_info = {
                'file_id': f"{delta_id}_metadata",
                'filename': f'delta_{delta_id}_metadata.json',
                'custom_name': None,
                'file_type': 'delta_metadata',
                'file_source': 'delta',
                'total_rows': 1,  # Metadata is just one record
                'total_columns': 0,
                'columns': [],
                'upload_time': timestamp_iso,
                'file_size_mb': 0,
                'data_types': {},
                'delta_id': delta_id,
                'result_type': 'metadata'
            }
            
            # Create metadata DataFrame with summary information
            metadata_df = pd.DataFrame([{
                'delta_id': delta_id,
                'timestamp': timestamp_iso,
                'row_counts': {
                    'unchanged': unchanged_count,
                    'amended': amended_count,
                    'deleted': deleted_count,
                    'newly_added': newly_added_count
                },
                'total_records': total_records
            }])
            
            metadata_data = {
                'info': metadata_info,
                'data': metadata_df
            }
            
            success = self.storage.save(f"{delta_id}_metadata", metadata_data, metadata_info)
            all_success = all_success and success
            if success:
                logger.info(f"DeltaStorage: Stored metadata")
            
            elapsed = time.time() - start_time
            if all_success:
                logger.info(f"DeltaStorage STORE SUCCESS: {delta_id} stored as separate files in uploaded_files in {elapsed:.3f}s")
            else:
                logger.error(f"DeltaStorage STORE FAILED: {delta_id} failed after {elapsed:.3f}s")
            
            return all_success
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"DeltaStorage STORE ERROR: {delta_id} failed after {elapsed:.3f}s - {e}")
            return False

    def get_results(self, delta_id: str) -> Optional[Dict]:
        """Get full stored results by combining separate files"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"DeltaStorage RETRIEVE: {delta_id}")
            
            # Try to get the separate files created by the new storage format
            result = {
                'unchanged': [],
                'amended': [],
                'deleted': [],
                'newly_added': [],
                'all_changes': [],
                'row_counts': {'unchanged': 0, 'amended': 0, 'deleted': 0, 'newly_added': 0},
                'timestamp': None
            }
            
            total_records = 0
            
            # Get unchanged results
            if self.storage.exists(f"{delta_id}_unchanged"):
                unchanged_data = self.storage.get(f"{delta_id}_unchanged")
                result['unchanged'] = unchanged_data['data'].to_dict('records')
                result['row_counts']['unchanged'] = len(result['unchanged'])
                result['timestamp'] = unchanged_data['info'].get('upload_time')
                total_records += len(result['unchanged'])
            
            # Get amended results
            if self.storage.exists(f"{delta_id}_amended"):
                amended_data = self.storage.get(f"{delta_id}_amended")
                result['amended'] = amended_data['data'].to_dict('records')
                result['row_counts']['amended'] = len(result['amended'])
                if not result['timestamp']:
                    result['timestamp'] = amended_data['info'].get('upload_time')
                total_records += len(result['amended'])
            
            # Get deleted results
            if self.storage.exists(f"{delta_id}_deleted"):
                deleted_data = self.storage.get(f"{delta_id}_deleted")
                result['deleted'] = deleted_data['data'].to_dict('records')
                result['row_counts']['deleted'] = len(result['deleted'])
                if not result['timestamp']:
                    result['timestamp'] = deleted_data['info'].get('upload_time')
                total_records += len(result['deleted'])
            
            # Get newly_added results
            if self.storage.exists(f"{delta_id}_newly_added"):
                newly_added_data = self.storage.get(f"{delta_id}_newly_added")
                result['newly_added'] = newly_added_data['data'].to_dict('records')
                result['row_counts']['newly_added'] = len(result['newly_added'])
                if not result['timestamp']:
                    result['timestamp'] = newly_added_data['info'].get('upload_time')
                total_records += len(result['newly_added'])
            
            # Create all_changes by combining amended, deleted, newly_added
            result['all_changes'] = result['amended'] + result['deleted'] + result['newly_added']
            
            # Convert timestamp string back to datetime if needed
            if result['timestamp'] and isinstance(result['timestamp'], str):
                from datetime import datetime
                try:
                    result['timestamp'] = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                except:
                    result['timestamp'] = datetime.utcnow()
            
            elapsed = time.time() - start_time
            logger.info(f"DeltaStorage RETRIEVE SUCCESS: {delta_id} retrieved {total_records} records in {elapsed:.3f}s")
            
            return result if total_records > 0 else None
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"DeltaStorage RETRIEVE ERROR: {delta_id} failed after {elapsed:.3f}s - {e}")
            return None

    def get_metadata_only(self, delta_id: str) -> Optional[Dict]:
        """Get only metadata for quick access without loading full data"""
        import time
        start_time = time.time()
        
        try:
            logger.debug(f"DeltaStorage METADATA: {delta_id}")
            
            row_counts = {'unchanged': 0, 'amended': 0, 'deleted': 0, 'newly_added': 0}
            timestamp = None
            
            # Get row counts from each file's metadata without loading data
            if self.storage.exists(f"{delta_id}_unchanged"):
                unchanged_metadata = self.storage.get_metadata_only(f"{delta_id}_unchanged")
                if unchanged_metadata:
                    row_counts['unchanged'] = unchanged_metadata.get('total_rows', 0)
                    timestamp = unchanged_metadata.get('upload_time')
            
            if self.storage.exists(f"{delta_id}_amended"):
                amended_metadata = self.storage.get_metadata_only(f"{delta_id}_amended")
                if amended_metadata:
                    row_counts['amended'] = amended_metadata.get('total_rows', 0)
                    if not timestamp:
                        timestamp = amended_metadata.get('upload_time')
            
            if self.storage.exists(f"{delta_id}_deleted"):
                deleted_metadata = self.storage.get_metadata_only(f"{delta_id}_deleted")
                if deleted_metadata:
                    row_counts['deleted'] = deleted_metadata.get('total_rows', 0)
                    if not timestamp:
                        timestamp = deleted_metadata.get('upload_time')
            
            if self.storage.exists(f"{delta_id}_newly_added"):
                newly_added_metadata = self.storage.get_metadata_only(f"{delta_id}_newly_added")
                if newly_added_metadata:
                    row_counts['newly_added'] = newly_added_metadata.get('total_rows', 0)
                    if not timestamp:
                        timestamp = newly_added_metadata.get('upload_time')
            
            total_records = sum(row_counts.values())
            
            if total_records > 0:
                elapsed = time.time() - start_time
                result = {
                    'delta_id': delta_id,
                    'row_counts': row_counts,
                    'timestamp': timestamp,
                    'total_records': total_records
                }
                logger.debug(f"DeltaStorage METADATA SUCCESS: {delta_id} in {elapsed:.3f}s - {total_records} records")
                return result
            else:
                logger.debug(f"DeltaStorage METADATA NOT FOUND: {delta_id}")
                return None
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"DeltaStorage METADATA ERROR: {delta_id} failed after {elapsed:.3f}s - {e}")
            return None

    def delete_results(self, delta_id: str) -> bool:
        """Delete all files related to a delta result"""
        try:
            logger.info(f"DeltaStorage DELETE: {delta_id}")
            
            success_count = 0
            total_count = 0
            
            # List of all possible file types for a delta result
            file_types = ['unchanged', 'amended', 'deleted', 'newly_added', 'metadata']
            
            for file_type in file_types:
                file_id = f"{delta_id}_{file_type}"
                total_count += 1
                
                if self.storage.exists(file_id):
                    try:
                        success = self.storage.delete(file_id)
                        if success:
                            success_count += 1
                            logger.info(f"DeltaStorage: Deleted {file_type} file for {delta_id}")
                        else:
                            logger.warning(f"DeltaStorage: Failed to delete {file_type} file for {delta_id}")
                    except Exception as e:
                        logger.warning(f"DeltaStorage: Error deleting {file_type} file for {delta_id}: {e}")
                else:
                    # File doesn't exist, which is fine
                    success_count += 1
            
            success = success_count == total_count
            if success:
                logger.info(f"DeltaStorage DELETE SUCCESS: {delta_id} - all files deleted")
            else:
                logger.warning(f"DeltaStorage DELETE PARTIAL: {delta_id} - {success_count}/{total_count} files deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"DeltaStorage DELETE ERROR: {delta_id} - {e}")
            return False

    def list_deltas(self, limit: int = 50) -> List[str]:
        """List all available delta IDs by looking for metadata files"""
        try:
            all_files = self.storage.list()
            delta_ids = []
            
            for file_id in all_files:
                if file_id.startswith('delta_') and '_metadata' in file_id:
                    # Extract delta ID from metadata file name
                    delta_id = file_id.replace('_metadata', '')
                    if delta_id not in delta_ids:
                        delta_ids.append(delta_id)
            
            # Sort by most recent (assuming delta IDs contain timestamps)
            delta_ids.sort(reverse=True)
            
            return delta_ids[:limit]
            
        except Exception as e:
            logger.error(f"DeltaStorage LIST ERROR: {e}")
            return []


# Create global instance
optimized_delta_storage = OptimizedDeltaStorage()