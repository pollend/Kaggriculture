$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ContainerName = "kaggriculture-postgres"
$VolumeName = "kaggriculture-postgres-data"
$ImageName = "postgres:17-alpine"
$LegacyContainerName = "kaggriculture-postgres-1"

function Test-LabContainer {
    docker container inspect $ContainerName *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-LabVolume {
    if (Test-LabContainer) {
        $MountedVolume = docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' $ContainerName
        if ($MountedVolume) {
            $script:VolumeName = $MountedVolume
        }
    }
}

function Start-LabPostgres {
    if (-not (Test-LabContainer)) {
        docker container inspect $LegacyContainerName *> $null
        if ($LASTEXITCODE -eq 0) {
            $LegacyVolume = docker inspect --format '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' $LegacyContainerName
            if ($LegacyVolume) {
                $script:VolumeName = $LegacyVolume
            }
            docker stop $LegacyContainerName *> $null
            docker rm $LegacyContainerName | Out-Null
        }
    }
    docker volume create $VolumeName | Out-Null
    if (Test-LabContainer) {
        $Running = docker inspect --format "{{.State.Running}}" $ContainerName
        if ($Running -ne "true") {
            docker start $ContainerName | Out-Null
        }
    }
    else {
        docker run -d `
            --name $ContainerName `
            --restart unless-stopped `
            -e POSTGRES_DB=kaggriculture `
            -e POSTGRES_USER=kag_admin `
            -e POSTGRES_PASSWORD=kag_admin_local `
            -p 127.0.0.1:54329:5432 `
            -v "${VolumeName}:/var/lib/postgresql/data" `
            $ImageName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the PostgreSQL container."
        }
    }

    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        docker exec $ContainerName pg_isready -U kag_admin -d kaggriculture *> $null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "PostgreSQL did not become ready within 60 seconds."
}

$CommandName = if ($args.Count -gt 0) { $args[0] } else { "up" }
Push-Location $RepoRoot
try {
    Resolve-LabVolume
    switch ($CommandName) {
        "up" {
            uv run python -m kaggriculture_lab.postgres_admin write-env
            if ($LASTEXITCODE -ne 0) { throw "Could not write the PostgreSQL environment file." }
            Start-LabPostgres
            uv run python -m kaggriculture_lab.postgres_admin bootstrap
            if ($LASTEXITCODE -ne 0) { throw "Could not bootstrap PostgreSQL." }
            uv run kaglab web
            exit $LASTEXITCODE
        }
        "status" {
            docker ps -a --filter "name=^/$ContainerName$"
            exit $LASTEXITCODE
        }
        "logs" {
            docker logs -f $ContainerName
            exit $LASTEXITCODE
        }
        "down" {
            if (Test-LabContainer) {
                $Running = docker inspect --format "{{.State.Running}}" $ContainerName
                if ($Running -eq "true") {
                    docker stop $ContainerName | Out-Null
                }
            }
        }
        "reset" {
            if ($args.Count -lt 2 -or $args[1] -ne "--yes") {
                throw "reset deletes the PostgreSQL volume; rerun: ./lab-stack.ps1 reset --yes"
            }
            if (Test-LabContainer) {
                docker stop $ContainerName *> $null
                docker rm $ContainerName | Out-Null
            }
            docker volume rm $VolumeName
            exit $LASTEXITCODE
        }
        default {
            throw "usage: ./lab-stack.ps1 {up|status|logs|down|reset --yes}"
        }
    }
}
finally {
    Pop-Location
}
