@{
  Rules = @{
    # This is a style rule and has produced false positives in this repo.
    # Disabling it keeps VS Code Problems focused on actionable issues.
    PSUseApprovedVerbs = @{ Enable = $false }
  }
}
