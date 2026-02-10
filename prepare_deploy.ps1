$source = "C:\Users\86159\Desktop\collision-model"
$dest = "C:\Users\86159\Desktop\collision_model_deploy"

# Create destination
if (Test-Path $dest) {
    Remove-Item $dest -Recurse -Force
}
New-Item -ItemType Directory -Path $dest | Out-Null

# Copy root files
$rootFiles = @("package.xml", "CMakeLists.txt", "requirements_robot.txt")
foreach ($file in $rootFiles) {
    Copy-Item -Path "$source\$file" -Destination "$dest\$file"
}

# Copy launch
Copy-Item -Path "$source\launch" -Destination "$dest" -Recurse

# Copy scripts (excluding pycache)
New-Item -ItemType Directory -Path "$dest\scripts" | Out-Null
Get-ChildItem -Path "$source\scripts" | Where-Object { $_.Name -ne "__pycache__" } | Copy-Item -Destination "$dest\scripts" -Recurse

# Copy checkpoints (excluding .pth)
New-Item -ItemType Directory -Path "$dest\checkpoints" | Out-Null
Get-ChildItem -Path "$source\checkpoints" | Where-Object { $_.Extension -ne ".pth" } | Copy-Item -Destination "$dest\checkpoints" -Recurse

Write-Host "Deployment package created at $dest"
