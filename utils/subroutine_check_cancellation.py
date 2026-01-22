class TaskCancelledError(Exception):
    """Custom exception to signal task cancellation."""
    pass

def check_cancellation(user_id, active_tasks):
    """Check if task has been cancelled and raise exception if it has."""
    print(f"DEBUG: Checking cancellation for user {user_id}")
    print(f"DEBUG: Active tasks keys: {list(active_tasks.keys()) if active_tasks else 'None'}")
    
    if user_id and active_tasks:
        task_info = active_tasks.get(user_id, {})
        current_status = task_info.get('status', 'unknown')
        print(f"DEBUG: Current status for user {user_id}: {current_status}")
        
        if current_status == 'cancelling':
            print(f"DEBUG: CANCELLATION DETECTED for user {user_id}")
            raise TaskCancelledError("Task was cancelled by the user.")
    else:
        print(f"DEBUG: No cancellation check possible - user_id: {user_id}, active_tasks: {active_tasks is not None}") 