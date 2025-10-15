import sys
import traceback
from typing import Optional, cast


class DocumentPortalException(Exception):
    def __init__(self, error_message, original_execption: Optional[object] = None):
        # Normalize message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Resolve exc_info (supports: sys module, Exception object, or current context)
        exc_type = exc_value = exc_tb = None
        if original_execption is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(original_execption, "exc_info"):  # e.g., sys
                #exc_type, exc_value, exc_tb = original_execption.exc_info()
                exc_info_obj = cast(sys, original_execption)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(original_execption, BaseException):
                exc_type, exc_value, exc_tb = type(original_execption), original_execption, original_execption.__traceback__
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the last frame to report the most relevant location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Full pretty traceback (if available)
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        # Compact, logger-friendly message (no leading spaces)
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"