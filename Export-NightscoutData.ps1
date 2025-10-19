param(
    [Parameter()]
    [string]$FromDate = (Get-Date).AddMonths(-1).ToString("yyyy-MM-dd"),

    [Parameter()]
    [string]$ToDate = $null,

    [Parameter()]
    [int]$MaxCount = 100000,

    [Parameter()]
    [bool]$UseLocalTimezone = $true,

    [Parameter()]
    [bool]$ConvertToMmol = $true,

    [Parameter()]
    [string]$OutputFolder = "./Output"
)

# Load environment variables from .env file
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

# Get API credentials from environment variables
$token = $env:NIGHTSCOUT_TOKEN
$baseUrl = $env:NIGHTSCOUT_URL

if (-not $token -or -not $baseUrl) {
    Write-Error "Missing required environment variables. Please ensure .env file contains NIGHTSCOUT_TOKEN and NIGHTSCOUT_URL"
    exit 1
}

# Create headers
$headers = @{
    'accept' = 'application/json'
}

# Resolve output folder path (relative to script root if not absolute)
if ([System.IO.Path]::IsPathRooted($OutputFolder)) {
    $outputFolder = $OutputFolder
}
else {
    $outputFolder = Join-Path $PSScriptRoot $OutputFolder
}

# Create output folder if it doesn't exist
if (!(Test-Path $outputFolder)) {
    New-Item -ItemType Directory -Path $outputFolder | Out-Null
    Write-Host "Created output folder: $outputFolder"
}

# Parse and validate date parameters
try {
    $fromDateTime = [DateTime]::ParseExact($FromDate, "yyyy-MM-dd", $null)
}
catch {
    Write-Error "Invalid FromDate format. Please use yyyy-MM-dd format (e.g., 2025-01-01)"
    exit 1
}

if ($ToDate) {
    try {
        $toDateTime = [DateTime]::ParseExact($ToDate, "yyyy-MM-dd", $null)
    }
    catch {
        Write-Error "Invalid ToDate format. Please use yyyy-MM-dd format (e.g., 2025-12-31)"
        exit 1
    }
}
else {
    $toDateTime = [DateTime]::MaxValue
}

# Create date range suffix for file names (convert yyyy-MM-dd to yyyyMMdd format)
$fromDateFormatted = $FromDate -replace '-', ''
if ($ToDate) {
    $toDateFormatted = $ToDate -replace '-', ''
}
else {
    # Use current date if no end date specified
    $toDateFormatted = (Get-Date).ToString("yyyyMMdd")
}
$dateRangeSuffix = "$fromDateFormatted-$toDateFormatted"

Write-Host "Date range: From $FromDate" $(if ($ToDate) { "To $ToDate" } else { "(no upper limit)" })
Write-Host "Use local timezone: $UseLocalTimezone"
Write-Host "Convert to mmol/L: $ConvertToMmol"

# Prepare API query dates
if ($UseLocalTimezone) {
    # User's input dates are in local time, convert to UTC for API query
    # e.g., "2025-06-01 00:00:00 local" -> "2025-05-31 22:00:00 UTC" (if timezone is UTC+2)
    $utcFromDateTime = [System.TimeZoneInfo]::ConvertTimeToUtc($fromDateTime, [System.TimeZoneInfo]::Local)
    $apiFromDate = $utcFromDateTime.ToString("yyyy-MM-ddTHH:mm:ss")

    if ($ToDate) {
        # Convert end date (end of day in local time)
        $utcToDateTime = [System.TimeZoneInfo]::ConvertTimeToUtc($toDateTime.AddDays(1).AddSeconds(-1), [System.TimeZoneInfo]::Local)
        $apiToDate = $utcToDateTime.ToString("yyyy-MM-ddTHH:mm:ss")
    } else {
        $apiToDate = $null
    }

    Write-Host "API query (UTC): From $apiFromDate" $(if ($apiToDate) { "To $apiToDate" } else { "" })
} else {
    # User's input dates are already in UTC
    $apiFromDate = $FromDate
    $apiToDate = $ToDate
}

try {
    # Fetch entries data
    Write-Host "Fetching entries from Nightscout API (max $MaxCount records)..."
    $entriesUrl = "$baseUrl/api/v1/entries?count=$MaxCount&token=$token" + "&find[dateString][`$gte]=$apiFromDate"
    if ($apiToDate) {
        $entriesUrl += "&find[dateString][`$lte]=$apiToDate"
    }
    $entriesResponse = Invoke-RestMethod -Uri $entriesUrl -Headers $headers -Method Get

    # Convert timestamps and SGV values if requested
    foreach ($entry in $entriesResponse) {
        if ($UseLocalTimezone) {
            if ($entry.dateString) {
                $utcDate = [DateTime]::Parse($entry.dateString)
                $localDate = [System.TimeZoneInfo]::ConvertTimeFromUtc($utcDate, [System.TimeZoneInfo]::Local)
                $entry.dateString = $localDate.ToString("yyyy-MM-dd HH:mm:ss")
            }
            if ($entry.sysTime) {
                $utcDate = [DateTime]::Parse($entry.sysTime)
                $localDate = [System.TimeZoneInfo]::ConvertTimeFromUtc($utcDate, [System.TimeZoneInfo]::Local)
                $entry.sysTime = $localDate.ToString("yyyy-MM-dd HH:mm:ss")
            }
        }

        # Convert SGV from mg/dL to mmol/L if requested
        if ($ConvertToMmol -and $entry.sgv) {
            $entry.sgv = [math]::Round($entry.sgv / 18.0182, 1)
        }
    }

    # Sort entries by dateString in ascending order and save to CSV
    $entriesFile = Join-Path $outputFolder "$dateRangeSuffix-entries.csv"
    $sortedEntries = $entriesResponse | Sort-Object { [DateTime]$_.dateString }
    $sortedEntries | Export-Csv -Path $entriesFile -NoTypeInformation
    Write-Host "Entries exported to: $entriesFile"
    Write-Host "Total entries: $($entriesResponse.Count)"

    # Fetch treatments data
    Write-Host "`nFetching treatments from Nightscout API (max $MaxCount records)..."
    $treatmentsUrl = "$baseUrl/api/v1/treatments?count=$MaxCount&token=$token" + "&find[created_at][`$gte]=$apiFromDate"
    if ($apiToDate) {
        $treatmentsUrl += "&find[created_at][`$lte]=$apiToDate"
    }
    $treatmentsResponse = Invoke-RestMethod -Uri $treatmentsUrl -Headers $headers -Method Get

    # Normalize treatments and convert timestamps
    $normalizedTreatments = [System.Collections.ArrayList]::new()
    foreach ($treatment in $treatmentsResponse) {
        # Ensure both insulin and amount properties exist
        if (-not ($treatment.PSObject.Properties.Name -contains 'insulin')) {
            $treatment | Add-Member -NotePropertyName 'insulin' -NotePropertyValue $null
        }
        if (-not ($treatment.PSObject.Properties.Name -contains 'amount')) {
            $treatment | Add-Member -NotePropertyName 'amount' -NotePropertyValue $null
        }

        # Convert timestamps if requested
        if ($UseLocalTimezone) {
            if ($treatment.created_at) {
                $utcDate = [DateTime]::Parse($treatment.created_at)
                $localDate = [System.TimeZoneInfo]::ConvertTimeFromUtc($utcDate, [System.TimeZoneInfo]::Local)
                $treatment.created_at = $localDate.ToString("yyyy-MM-dd HH:mm:ss")
            }
            if ($treatment.timestamp) {
                $utcDate = [DateTime]::Parse($treatment.timestamp)
                $localDate = [System.TimeZoneInfo]::ConvertTimeFromUtc($utcDate, [System.TimeZoneInfo]::Local)
                $treatment.timestamp = $localDate.ToString("yyyy-MM-dd HH:mm:ss")
            }
        }

        [void]$normalizedTreatments.Add($treatment)
    }

    # Sort treatments by created_at in ascending order and save to CSV (includes both insulin and amount columns)
    $treatmentsFile = Join-Path $outputFolder "$dateRangeSuffix-treatments.csv"
    $sortedTreatments = $normalizedTreatments | Sort-Object { [DateTime]$_.created_at }
    $sortedTreatments | Export-Csv -Path $treatmentsFile -NoTypeInformation
    Write-Host "Treatments exported to: $treatmentsFile"
    Write-Host "Total treatments: $($treatmentsResponse.Count)"

    # Process and combine datasets
    Write-Host "`nProcessing and combining datasets..."

    # Process entries - convert to desired format
    $processedEntries = [System.Collections.ArrayList]::new()
    foreach ($entry in $entriesResponse) {
        if ($entry.sgv) {
            # Parse the already-converted date string (conversion happened earlier if ConvertToLocal was true)
            $entryDate = [DateTime]::Parse($entry.dateString)

            # Apply date filtering on processed date
            if ($entryDate -ge $fromDateTime -and ($toDateTime -eq [DateTime]::MaxValue -or $entryDate -le $toDateTime)) {
                # Use the already-converted SGV value (conversion happened earlier if ConvertToMmol was true)
                [void]$processedEntries.Add([PSCustomObject]@{
                        DateTime  = $entryDate.ToString("yyyy-MM-dd HH:mm:ss")
                        eventType = "SGV"
                        sgv       = $entry.sgv
                        insulin   = $null
                        amount    = $null
                        carbs     = $null
                    })
            }
        }
    }

    # Process treatments - convert to desired format
    $processedTreatments = [System.Collections.ArrayList]::new()
    foreach ($treatment in $treatmentsResponse) {
        # Parse the already-converted date string (conversion happened earlier if ConvertToLocal was true)
        $treatmentDate = [DateTime]::Parse($treatment.created_at)

        # Apply date filtering on processed date
        if ($treatmentDate -ge $fromDateTime -and ($toDateTime -eq [DateTime]::MaxValue -or $treatmentDate -le $toDateTime)) {
            [void]$processedTreatments.Add([PSCustomObject]@{
                    DateTime  = $treatmentDate.ToString("yyyy-MM-dd HH:mm:ss")
                    eventType = $treatment.eventType
                    sgv       = $null
                    insulin   = $treatment.insulin
                    amount    = $treatment.amount
                    carbs     = $treatment.carbs
                })
        }
    }

    # Combine both datasets
    $combinedData = $processedEntries + $processedTreatments

    # Sort by DateTime
    $sortedData = $combinedData | Sort-Object { [DateTime]$_.DateTime }

    # Export combined data to CSV
    $combinedFile = Join-Path $outputFolder "$dateRangeSuffix-combined.csv"
    $sortedData | Export-Csv -Path $combinedFile -NoTypeInformation

    Write-Host "`nCombined data exported to: $combinedFile"
    Write-Host "Total SGV entries: $($processedEntries.Count)"
    Write-Host "Total treatments: $($processedTreatments.Count)"
    Write-Host "Total combined records: $($sortedData.Count)"

}
catch {
    Write-Error "Failed to retrieve data: $_"
    exit 1
}
