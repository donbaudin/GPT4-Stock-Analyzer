from typing import Dict, Any, Optional
import os
import csv
import pandas as pd
from datetime import datetime
from weasyprint import HTML
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exporter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Exporter:
    """
    Handles exporting analysis results to various formats.
    Complies with Twitter's API terms regarding data export and privacy.
    """
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize exporter with output directory.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different export types
        self.csv_dir = self.output_dir / "csv"
        self.excel_dir = self.output_dir / "excel"
        self.html_dir = self.output_dir / "html"
        self.pdf_dir = self.output_dir / "pdf"
        
        for directory in [self.csv_dir, self.excel_dir, self.html_dir, self.pdf_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Initialized exporter with output directory: {output_dir}")

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data to remove sensitive or raw content.
        """
        try:
            sanitized = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    sanitized[key] = self._sanitize_data(value)
                elif isinstance(value, list):
                    sanitized[key] = [
                        self._sanitize_data(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                elif key in ['text', 'content', 'raw_data', 'tweet_text', 'post_content']:
                    # Replace raw content with metadata
                    sanitized[key] = f"[Content length: {len(str(value))} characters]"
                else:
                    sanitized[key] = value
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing data: {str(e)}")
            return data

    def _get_timestamped_filename(self, base_name: str, extension: str) -> str:
        """
        Generate a timestamped filename.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"

    def export_to_csv(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to CSV format.
        Only includes aggregated metrics, not raw content.
        """
        try:
            sanitized_data = self._sanitize_data(data)
            output_file = self.csv_dir / self._get_timestamped_filename(filename, "csv")
            
            # Flatten nested dictionaries for CSV
            flattened_data = []
            for key, value in sanitized_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened_data.append({
                            "metric": f"{key}_{subkey}",
                            "value": subvalue
                        })
                else:
                    flattened_data.append({
                        "metric": key,
                        "value": value
                    })
            
            # Write to CSV
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "value"])
                writer.writeheader()
                writer.writerows(flattened_data)
            
            logger.info(f"Successfully exported data to CSV: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise

    def export_to_excel(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to Excel format.
        Only includes aggregated metrics, not raw content.
        """
        try:
            sanitized_data = self._sanitize_data(data)
            output_file = self.excel_dir / self._get_timestamped_filename(filename, "xlsx")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(sanitized_data, orient='index')
            df.index.name = 'metric'
            
            # Write to Excel
            df.to_excel(output_file)
            logger.info(f"Successfully exported data to Excel: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            raise

    def export_to_html(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to HTML format.
        Only includes aggregated metrics, not raw content.
        """
        try:
            sanitized_data = self._sanitize_data(data)
            output_file = self.html_dir / self._get_timestamped_filename(filename, "html")
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Stock Analysis Export</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; }}
                    .value {{ color: #0066cc; }}
                    .timestamp {{ color: #666; font-size: 0.8em; }}
                </style>
            </head>
            <body>
                <h1>Stock Analysis Export</h1>
                <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            """
            
            # Add metrics
            for key, value in sanitized_data.items():
                if isinstance(value, dict):
                    html_content += f"<h2>{key}</h2>"
                    for subkey, subvalue in value.items():
                        html_content += f"""
                        <div class="metric">
                            <strong>{subkey}:</strong>
                            <span class="value">{subvalue}</span>
                        </div>
                        """
                else:
                    html_content += f"""
                    <div class="metric">
                        <strong>{key}:</strong>
                        <span class="value">{value}</span>
                    </div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Successfully exported data to HTML: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to HTML: {str(e)}")
            raise

    def export_to_pdf(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to PDF format.
        Only includes aggregated metrics, not raw content.
        """
        try:
            # First create HTML
            html_file = self.export_to_html(data, filename)
            output_file = self.pdf_dir / self._get_timestamped_filename(filename, "pdf")
            
            # Convert HTML to PDF
            HTML(html_file).write_pdf(output_file)
            logger.info(f"Successfully exported data to PDF: {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            raise

    def export_summary(self, data: Dict[str, Any]) -> str:
        """
        Return a summary-only string for quick output.
        Only includes aggregated metrics, not raw content.
        """
        try:
            sanitized_data = self._sanitize_data(data)
            summary = []
            
            for key, value in sanitized_data.items():
                if isinstance(value, dict):
                    summary.append(f"{key}:")
                    for subkey, subvalue in value.items():
                        summary.append(f"  {subkey}: {subvalue}")
                else:
                    summary.append(f"{key}: {value}")
            
            logger.info("Successfully generated summary")
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
