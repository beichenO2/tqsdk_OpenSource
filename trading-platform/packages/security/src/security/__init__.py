"""Trading platform security package — credentials, encryption, auth, and log sanitization."""

from security.apikeys import APIKeyManager, APIKeyRecord
from security.config import (
    SecuritySettings,
    scan_dotenv_for_exposed_secrets,
    validate_env_file_for_secrets,
    verify_gitignore_covers_sensitive_files,
)
from security.encryption import (
    EncryptionService,
    FieldEncryptor,
    decrypt,
    derive_key_from_password,
    encrypt,
)
from security.jwt import JWTError, JWTService
from security.keychain import delete_credential, get_credential, store_credential
from security.middleware import (
    SanitizingLoggingMiddleware,
    add_cors,
    add_sanitizing_logging_middleware,
    cors_middleware_config,
)
from security.password import hash_password, verify_password
from security.sanitizer import LogSanitizer, sanitize, sanitize_log
# 260505 refactor removed AsyncPrivPortalClient/create_exchange_credentials
from security.privportal import (
    ExchangeKeys,
    PrivPortalClient,
    TqSdkKeys,
)
from security.vault import CredentialEntry, CredentialVault

__all__ = [
    "APIKeyManager",
    "APIKeyRecord",
    "CredentialEntry",
    "CredentialVault",
    "EncryptionService",
    "FieldEncryptor",
    "JWTError",
    "JWTService",
    "LogSanitizer",
    "SanitizingLoggingMiddleware",
    "SecuritySettings",
    "add_cors",
    "add_sanitizing_logging_middleware",
    "cors_middleware_config",
    "decrypt",
    "delete_credential",
    "derive_key_from_password",
    "encrypt",
    "get_credential",
    "hash_password",
    "sanitize",
    "sanitize_log",
    "scan_dotenv_for_exposed_secrets",
    "store_credential",
    "validate_env_file_for_secrets",
    "verify_gitignore_covers_sensitive_files",
    "verify_password",
    "ExchangeKeys",
    "PrivPortalClient",
    "TqSdkKeys",
]
