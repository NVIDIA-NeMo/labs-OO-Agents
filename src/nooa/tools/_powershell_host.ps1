# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Commands and replies use a private control socket. Standard handles belong to
# user commands, so neither native stdin nor malformed syntax can eat protocol.
param([int]$ControlPort, [string]$ControlToken)

$__nooa_utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $__nooa_utf8
[Console]::OutputEncoding = $__nooa_utf8
$OutputEncoding = $__nooa_utf8
$ProgressPreference = 'SilentlyContinue'
if ($PSVersionTable.PSVersion -ge [version]'7.2') { $PSStyle.OutputRendering = 'PlainText' }

$__nooa_client = [System.Net.Sockets.TcpClient]::new('127.0.0.1', $ControlPort)
$__nooa_reader = [System.IO.StreamReader]::new($__nooa_client.GetStream(), $__nooa_utf8)
$__nooa_writer = [System.IO.StreamWriter]::new($__nooa_client.GetStream(), $__nooa_utf8)
$__nooa_writer.NewLine = "`n"
$__nooa_writer.AutoFlush = $true
$__nooa_writer.WriteLine($ControlToken)
# Python assigns this process to its job before allowing setup or user code.
if ($__nooa_reader.ReadLine() -ne 'start') { exit 1 }

try {
    # Windows PowerShell 5.1 can fall back to its installation directory when
    # the inherited cwd contains wildcard characters. The native process cwd
    # still holds the requested literal path; initialize PowerShell from it.
    Set-Location -LiteralPath ([Environment]::CurrentDirectory) -ErrorAction Stop
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class NooaStandardInput {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr GetStdHandle(int which);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetStdHandle(int which, IntPtr handle);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
}
'@ -ErrorAction Stop
    $__nooa_default_input = [NooaStandardInput]::GetStdHandle(-10)
    $__nooa_writer.WriteLine('{"ready":true}')
} catch {
    $__nooa_writer.WriteLine((@{error = $_.ToString()} | ConvertTo-Json -Compress))
    exit 1
}

while ($null -ne ($__nooa_line = $__nooa_reader.ReadLine())) {
    $__nooa_request = $__nooa_line | ConvertFrom-Json
    $__nooa_code = 0
    $__nooa_back = Get-Location
    $__nooa_input_file = $null
    $__nooa_input_reader = $null
    $__nooa_success = $true
    $__nooa_native_exit = 0
    $__nooa_previous_error = if ($Error.Count -gt 0) { $Error[0] } else { $null }
    $__nooa_last_error = $null
    $global:LASTEXITCODE = 0
    try {
        if ($null -ne $__nooa_request.cwd) {
            # In 5.1, even -LiteralPath can retain wildcard escapes while
            # normalizing relative paths. Resolve first without globbing.
            $__nooa_cwd = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
                $__nooa_request.cwd)
            Set-Location -LiteralPath $__nooa_cwd -ErrorAction Stop
        }
        if ($null -ne $__nooa_request.stdin_path) {
            $__nooa_input_file = [System.IO.File]::Open(
                $__nooa_request.stdin_path, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
            $__nooa_input_handle = $__nooa_input_file.SafeFileHandle.DangerousGetHandle()
            if (-not [NooaStandardInput]::SetHandleInformation($__nooa_input_handle, 1, 1) -or
                -not [NooaStandardInput]::SetStdHandle(-10, $__nooa_input_handle)) {
                throw 'Could not redirect native standard input'
            }
            $__nooa_input_reader = [System.IO.StreamReader]::new($__nooa_input_file, $__nooa_utf8)
            [Console]::SetIn($__nooa_input_reader)
        } else {
            [Console]::SetIn([System.IO.TextReader]::Null)
        }
        # Dot sourcing keeps variables, functions, environment, and cd in this
        # session. Finally captures the *last command's* status, including return;
        # inspecting $? after Out-Default would incorrectly turn failures into 0.
        $__nooa_block = [scriptblock]::Create(
            "try {`n" + $__nooa_request.command + "`n} finally {`n" +
            '$script:__nooa_success = $?; $script:__nooa_native_exit = $global:LASTEXITCODE; $script:__nooa_last_error = if ($Error.Count -gt 0) { $Error[0] } else { $null }' + "`n}")
        # Leave stderr on its own handle. In Windows PowerShell 5.1, 2>&1
        # converts native diagnostics into terminating NativeCommandError
        # records when ErrorActionPreference is Stop, even for exit code 0.
        . $__nooa_block | Out-Default
        if (-not $__nooa_success) {
            $__nooa_code = 1
            if ($__nooa_native_exit -ne 0) { $__nooa_code = $__nooa_native_exit }
            elseif ($null -ne $__nooa_last_error -and
                -not [object]::ReferenceEquals($__nooa_previous_error, $__nooa_last_error) -and
                $__nooa_last_error.FullyQualifiedErrorId -in @('NativeCommandError', 'NativeCommandErrorMessage')) {
                # PowerShell 5.1 turns redirected native stderr into errors,
                # even when that process exits successfully. Preserve its exit
                # status while keeping the diagnostic on stderr.
                $__nooa_code = 0
            }
        }
    } catch {
        $__nooa_code = 1
        [Console]::Error.WriteLine($_.ToString())
    } finally {
        [void][NooaStandardInput]::SetStdHandle(-10, $__nooa_default_input)
        [Console]::SetIn([System.IO.TextReader]::Null)
        if ($null -ne $__nooa_input_reader) { $__nooa_input_reader.Dispose() }
        if ($null -ne $__nooa_input_file) { $__nooa_input_file.Dispose() }
        if ($null -ne $__nooa_request.cwd) {
            Set-Location -LiteralPath $__nooa_back.Path -ErrorAction Stop
        }
        [Console]::Out.Flush()
        [Console]::Error.Flush()
        # Each pipe confirms its own end, including when user code calls exit.
        # The control connection can close while a slow consumer is still
        # draining output, so these markers must run during exit's unwinding.
        $__nooa_marker = [char]30 + 'NOOA_' + $__nooa_request.id + [char]31
        [Console]::Out.Write($__nooa_marker)
        [Console]::Error.Write($__nooa_marker)
        [Console]::Out.Flush()
        [Console]::Error.Flush()
    }
    $__nooa_reply = @{
        id = $__nooa_request.id
        code = $__nooa_code
        cwd = (Get-Location).ProviderPath
    } | ConvertTo-Json -Compress
    $__nooa_writer.WriteLine($__nooa_reply)
}
