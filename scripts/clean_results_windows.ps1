Remove-Item -Recurse -Force results\neurreps_sha_test_s32_main -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force results\neurreps_sha_test_s32_capacity -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force results\neurreps_sha_test_s32_padding_free -ErrorAction SilentlyContinue
Write-Output "Removed V10 result directories. DATA/ was not touched."
