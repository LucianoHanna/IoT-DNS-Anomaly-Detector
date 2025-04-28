import re
import pandas as pd

class DNSLogParser:
    def parse_dnsmasq_log(self, log_file):
        """Parse DNS logs from dnsmasq"""
        domains = []
        timestamps = []
        
        with open(log_file, 'r') as f:
            for line in f:
                # Extract domain name and timestamp
                match = re.search(r'(\w+\s+\d+\s+\d+:\d+:\d+).*query\[\w+\]\s+([a-zA-Z0-9.-]+)', line)
                if match:
                    timestamp, domain = match.groups()
                    domains.append(domain)
                    timestamps.append(timestamp)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, format='%b %d %H:%M:%S', errors='coerce'),
            'domain': domains
        })
        
        return df