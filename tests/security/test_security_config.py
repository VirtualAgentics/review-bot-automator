"""Tests for security configuration module."""

import pytest

from review_bot_automator.security.config import SecurityConfig


class TestSecurityConfigDefaults:
    """Test default configuration values."""

    def test_default_input_validation_settings(self) -> None:
        """Test default input validation settings are secure."""
        config = SecurityConfig()

        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.allowed_extensions == frozenset(
            {".py", ".ts", ".js", ".json", ".yaml", ".yml", ".toml"}
        )
        assert config.enable_path_validation is True
        assert config.enable_content_sanitization is True

    def test_default_secret_scanning_settings(self) -> None:
        """Test default secret scanning settings are secure."""
        config = SecurityConfig()

        assert config.enable_secret_scanning is True
        assert config.scan_on_file_read is True
        assert config.block_on_secrets_found is False  # Warning only

    def test_default_file_operations_settings(self) -> None:
        """Test default file operations settings are secure."""
        config = SecurityConfig()

        assert config.enable_atomic_writes is True
        assert config.enable_backups is True
        assert config.max_backup_count == 5

    def test_default_github_api_settings(self) -> None:
        """Test default GitHub API settings."""
        config = SecurityConfig()

        assert config.github_api_timeout == 30
        assert config.github_max_retries == 3

    def test_default_logging_settings(self) -> None:
        """Test default logging settings are secure."""
        config = SecurityConfig()

        assert config.enable_audit_logging is True
        assert config.log_sensitive_data is False  # Never log secrets


class TestSecurityConfigProfiles:
    """Test preset security profiles."""

    def test_conservative_profile(self) -> None:
        """Test conservative profile has strict settings."""
        config = SecurityConfig.conservative()

        # Stricter than default
        assert config.max_file_size == 5 * 1024 * 1024  # 5MB
        assert config.block_on_secrets_found is True
        assert config.max_backup_count == 10
        assert config.github_api_timeout == 20

        # Still secure defaults
        assert config.enable_secret_scanning is True
        assert config.enable_atomic_writes is True
        assert config.log_sensitive_data is False

    def test_balanced_profile(self) -> None:
        """Test balanced profile uses default settings."""
        balanced = SecurityConfig.balanced()
        default = SecurityConfig()

        # Balanced should match defaults
        assert balanced.max_file_size == default.max_file_size
        assert balanced.allowed_extensions == default.allowed_extensions
        assert balanced.enable_secret_scanning == default.enable_secret_scanning
        assert balanced.block_on_secrets_found == default.block_on_secrets_found
        assert balanced.max_backup_count == default.max_backup_count
        assert balanced.github_api_timeout == default.github_api_timeout

    def test_permissive_profile(self) -> None:
        """Test permissive profile is relaxed for dev/CI."""
        config = SecurityConfig.permissive()

        # More relaxed than default
        assert config.max_file_size == 20 * 1024 * 1024  # 20MB
        assert config.enable_secret_scanning is False
        assert config.block_on_secrets_found is False
        assert config.max_backup_count == 3
        assert config.github_api_timeout == 60

        # Still has some security features
        assert config.enable_atomic_writes is True
        assert config.log_sensitive_data is False

    def test_profile_security_ordering(self) -> None:
        """Test profiles are ordered by security strictness."""
        conservative = SecurityConfig.conservative()
        balanced = SecurityConfig.balanced()
        permissive = SecurityConfig.permissive()

        # File size: conservative < balanced < permissive
        assert conservative.max_file_size < balanced.max_file_size
        assert balanced.max_file_size < permissive.max_file_size

        # Secret blocking: conservative most strict
        assert conservative.block_on_secrets_found is True
        assert balanced.block_on_secrets_found is False
        assert permissive.block_on_secrets_found is False

        # Secret scanning: permissive disables it
        assert conservative.enable_secret_scanning is True
        assert balanced.enable_secret_scanning is True
        assert permissive.enable_secret_scanning is False


class TestSecurityConfigImmutability:
    """Test dataclass immutability (frozen)."""

    def test_cannot_modify_config_attributes(self) -> None:
        """Test that config attributes cannot be modified after creation."""
        config = SecurityConfig()

        with pytest.raises(Exception, match=r"cannot assign to field|can't set attribute"):
            config.max_file_size = 999  # type: ignore[misc]

        with pytest.raises(Exception, match=r"cannot assign to field|can't set attribute"):
            config.enable_secret_scanning = False  # type: ignore[misc]

        with pytest.raises(Exception, match=r"cannot assign to field|can't set attribute"):
            config.github_api_timeout = 100  # type: ignore[misc]

    def test_frozen_dataclass_is_hashable(self) -> None:
        """Test that frozen dataclass instances are hashable."""
        config1 = SecurityConfig()
        config2 = SecurityConfig()

        # Should be hashable (can use in set/dict)
        config_set = {config1, config2}
        assert len(config_set) == 1  # Same config, same hash

        # Different configs should have different hashes
        config3 = SecurityConfig.conservative()
        config_set.add(config3)
        assert len(config_set) == 2


class TestSecurityConfigTypes:
    """Test correct types for all fields."""

    def test_allowed_extensions_is_frozenset(self) -> None:
        """Test allowed_extensions uses frozenset, not mutable set."""
        config = SecurityConfig()

        assert isinstance(config.allowed_extensions, frozenset)
        assert type(config.allowed_extensions) is frozenset

        # Verify it's truly immutable
        with pytest.raises(AttributeError):
            config.allowed_extensions.add(".txt")  # type: ignore[attr-defined]

    def test_all_boolean_fields_are_bool(self) -> None:
        """Test all boolean configuration fields are actually bool."""
        config = SecurityConfig()

        assert isinstance(config.enable_path_validation, bool)
        assert isinstance(config.enable_content_sanitization, bool)
        assert isinstance(config.enable_secret_scanning, bool)
        assert isinstance(config.scan_on_file_read, bool)
        assert isinstance(config.block_on_secrets_found, bool)
        assert isinstance(config.enable_atomic_writes, bool)
        assert isinstance(config.enable_backups, bool)
        assert isinstance(config.enable_audit_logging, bool)
        assert isinstance(config.log_sensitive_data, bool)

    def test_all_integer_fields_are_int(self) -> None:
        """Test all integer configuration fields are actually int."""
        config = SecurityConfig()

        assert isinstance(config.max_file_size, int)
        assert isinstance(config.max_backup_count, int)
        assert isinstance(config.github_api_timeout, int)
        assert isinstance(config.github_max_retries, int)


class TestSecurityConfigCustomization:
    """Test custom configuration creation."""

    def test_custom_file_size(self) -> None:
        """Test creating config with custom file size."""
        config = SecurityConfig(max_file_size=15 * 1024 * 1024)

        assert config.max_file_size == 15 * 1024 * 1024
        # Other settings remain default
        assert config.enable_secret_scanning is True
        assert config.github_api_timeout == 30

    def test_custom_extensions(self) -> None:
        """Test creating config with custom allowed extensions."""
        custom_extensions = frozenset({".py", ".json"})
        config = SecurityConfig(allowed_extensions=custom_extensions)

        assert config.allowed_extensions == custom_extensions
        assert len(config.allowed_extensions) == 2

    def test_disable_secret_scanning(self) -> None:
        """Test creating config with secret scanning disabled."""
        config = SecurityConfig(enable_secret_scanning=False)

        assert config.enable_secret_scanning is False
        # Other settings remain default
        assert config.max_file_size == 10 * 1024 * 1024

    def test_multiple_custom_settings(self) -> None:
        """Test creating config with multiple custom settings."""
        config = SecurityConfig(
            max_file_size=7 * 1024 * 1024,
            block_on_secrets_found=True,
            github_api_timeout=45,
            max_backup_count=8,
        )

        assert config.max_file_size == 7 * 1024 * 1024
        assert config.block_on_secrets_found is True
        assert config.github_api_timeout == 45
        assert config.max_backup_count == 8


class TestSecurityConfigEquality:
    """Test configuration equality and hashing."""

    def test_equal_configs_are_equal(self) -> None:
        """Test that configs with same values are equal."""
        config1 = SecurityConfig()
        config2 = SecurityConfig()

        assert config1 == config2
        assert hash(config1) == hash(config2)

    def test_different_configs_are_not_equal(self) -> None:
        """Test that configs with different values are not equal."""
        config1 = SecurityConfig()
        config2 = SecurityConfig.conservative()

        assert config1 != config2
        assert hash(config1) != hash(config2)

    def test_custom_config_equality(self) -> None:
        """Test equality with custom configs."""
        config1 = SecurityConfig(max_file_size=15 * 1024 * 1024)
        config2 = SecurityConfig(max_file_size=15 * 1024 * 1024)
        config3 = SecurityConfig(max_file_size=20 * 1024 * 1024)

        assert config1 == config2
        assert config1 != config3
        assert hash(config1) == hash(config2)


class TestSecurityConfigValidation:
    """Test validation error paths in __post_init__."""

    def test_invalid_max_file_size_zero(self) -> None:
        """Test that zero max_file_size raises ValueError."""
        with pytest.raises(ValueError, match="max_file_size must be a positive integer"):
            SecurityConfig(max_file_size=0)

    def test_invalid_max_file_size_negative(self) -> None:
        """Test that negative max_file_size raises ValueError."""
        with pytest.raises(ValueError, match="max_file_size must be a positive integer"):
            SecurityConfig(max_file_size=-1)

    def test_invalid_max_file_size_not_int(self) -> None:
        """Test that non-integer max_file_size raises ValueError."""
        with pytest.raises(ValueError, match="max_file_size must be a positive integer"):
            SecurityConfig(max_file_size=10.5)  # type: ignore[arg-type]

    def test_invalid_github_api_timeout_zero(self) -> None:
        """Test that zero github_api_timeout raises ValueError."""
        with pytest.raises(ValueError, match="github_api_timeout must be a positive integer"):
            SecurityConfig(github_api_timeout=0)

    def test_invalid_github_api_timeout_negative(self) -> None:
        """Test that negative github_api_timeout raises ValueError."""
        with pytest.raises(ValueError, match="github_api_timeout must be a positive integer"):
            SecurityConfig(github_api_timeout=-5)

    def test_invalid_github_api_timeout_not_int(self) -> None:
        """Test that non-integer github_api_timeout raises ValueError."""
        with pytest.raises(ValueError, match="github_api_timeout must be a positive integer"):
            SecurityConfig(github_api_timeout=30.0)  # type: ignore[arg-type]

    def test_invalid_github_max_retries_zero(self) -> None:
        """Test that zero github_max_retries raises ValueError."""
        with pytest.raises(ValueError, match="github_max_retries must be a positive integer"):
            SecurityConfig(github_max_retries=0)

    def test_invalid_github_max_retries_negative(self) -> None:
        """Test that negative github_max_retries raises ValueError."""
        with pytest.raises(ValueError, match="github_max_retries must be a positive integer"):
            SecurityConfig(github_max_retries=-1)

    def test_invalid_github_max_retries_not_int(self) -> None:
        """Test that non-integer github_max_retries raises ValueError."""
        with pytest.raises(ValueError, match="github_max_retries must be a positive integer"):
            SecurityConfig(github_max_retries=3.5)  # type: ignore[arg-type]

    def test_invalid_max_backup_count_negative(self) -> None:
        """Test that negative max_backup_count raises ValueError."""
        with pytest.raises(ValueError, match="max_backup_count must be a non-negative integer"):
            SecurityConfig(max_backup_count=-1)

    def test_valid_max_backup_count_zero(self) -> None:
        """Test that zero max_backup_count is valid."""
        config = SecurityConfig(max_backup_count=0)
        assert config.max_backup_count == 0

    def test_invalid_max_backup_count_not_int(self) -> None:
        """Test that non-integer max_backup_count raises ValueError."""
        with pytest.raises(ValueError, match="max_backup_count must be a non-negative integer"):
            SecurityConfig(max_backup_count=5.5)  # type: ignore[arg-type]

    def test_invalid_allowed_extensions_not_frozenset(self) -> None:
        """Test that non-frozenset allowed_extensions raises ValueError."""
        with pytest.raises(ValueError, match="allowed_extensions must be a frozenset"):
            SecurityConfig(allowed_extensions={".py", ".js"})  # type: ignore[arg-type]

    def test_invalid_allowed_extensions_list(self) -> None:
        """Test that list allowed_extensions raises ValueError."""
        with pytest.raises(ValueError, match="allowed_extensions must be a frozenset"):
            SecurityConfig(allowed_extensions=[".py", ".js"])  # type: ignore[arg-type]

    def test_invalid_allowed_extensions_empty(self) -> None:
        """Test that empty allowed_extensions raises ValueError."""
        with pytest.raises(ValueError, match="allowed_extensions cannot be empty"):
            SecurityConfig(allowed_extensions=frozenset())

    def test_invalid_allowed_extensions_non_string(self) -> None:
        """Test that non-string elements in allowed_extensions raise ValueError."""
        with pytest.raises(ValueError, match="All extensions.*must be strings"):
            SecurityConfig(allowed_extensions=frozenset({".py", 123}))  # type: ignore[arg-type]

    def test_invalid_allowed_extensions_none_element(self) -> None:
        """Test that None element in allowed_extensions raises ValueError."""
        with pytest.raises(ValueError, match="All extensions.*must be strings"):
            SecurityConfig(allowed_extensions=frozenset({".py", None}))  # type: ignore[arg-type]
