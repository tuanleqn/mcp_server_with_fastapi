#!/usr/bin/env python3
"""
Port checker and killer utility for the MCP server project
"""

import subprocess
import sys
import re

def check_ports_in_use():
    """Check which ports are in use in the 8000-8010 range"""
    print("🔍 Checking ports 8000-8010...")
    
    try:
        # Run netstat to check ports
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        used_ports = []
        for line in lines:
            if ':80' in line and 'LISTENING' in line:
                # Extract port number
                match = re.search(r':(\d{4})', line)
                if match:
                    port = int(match.group(1))
                    if 8000 <= port <= 8010:
                        # Extract PID
                        pid_match = re.search(r'\s+(\d+)$', line.strip())
                        pid = pid_match.group(1) if pid_match else 'Unknown'
                        used_ports.append((port, pid))
        
        if used_ports:
            print("❌ Ports in use:")
            for port, pid in used_ports:
                print(f"   Port {port} - PID {pid}")
            return used_ports
        else:
            print("✅ No ports in range 8000-8010 are in use")
            return []
            
    except Exception as e:
        print(f"❌ Error checking ports: {e}")
        return []

def kill_process(pid):
    """Kill a process by PID"""
    try:
        subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=True)
        print(f"✅ Successfully killed process {pid}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to kill process {pid}")
        return False

def main():
    print("🔧 MCP Server Port Troubleshooter\n")
    
    used_ports = check_ports_in_use()
    
    if not used_ports:
        print("\n🎉 All ports are free! You can start the server now.")
        return
    
    print(f"\n💡 Found {len(used_ports)} port(s) in use.")
    print("\nOptions:")
    print("1. Kill all processes using ports 8000-8010")
    print("2. Kill specific process")
    print("3. Just show the information and exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🔥 Killing all processes using ports 8000-8010...")
            for port, pid in used_ports:
                print(f"Killing PID {pid} (port {port})...")
                kill_process(pid)
        
        elif choice == '2':
            pid = input("Enter the PID to kill: ").strip()
            if pid.isdigit():
                kill_process(int(pid))
            else:
                print("❌ Invalid PID")
        
        elif choice == '3':
            print("ℹ️  No action taken.")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
