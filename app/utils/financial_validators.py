# backend/app/utils/financial_validators.py
import logging
import re
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FinancialValidators:
    """
    Utility class for validating and extracting financial data
    """

    def __init__(self):
        self.patterns = {
            'ISIN': r'\b[A-Z]{2}[0-9A-Z]{10}\b',
            'CUSIP': r'\b[0-9A-Z]{9}\b',
            'SEDOL': r'\b[0-9A-Z]{7}\b',
            'amount': r'[\d,]+\.?\d*',
            'currency': r'\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|SGD|HKD|SEK|NOK|DKK|PLN|CZK|HUF|RUB|CNY|INR|KRW|THB|MXN|BRL|ZAR)\b',
            'date_patterns': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}\b'
            ],
            'trade_id': r'\b(TRD|TRADE|TX|REF)[0-9A-Z]+\b',
            'account': r'\b(ACC|ACCT|ACCOUNT)[0-9A-Z]+\b'
        }

        self.currency_symbols = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY',
            'kr': 'SEK', 'C$': 'CAD', 'A$': 'AUD'
        }

    def validate_isin(self, isin: str) -> bool:
        """
        Validate ISIN format and checksum
        """
        if not isin or len(isin) != 12:
            return False

        # Basic format check
        if not re.match(r'^[A-Z]{2}[0-9A-Z]{10}$', isin):
            return False

        # Check country code (first 2 characters)
        country_code = isin[:2]
        valid_countries = [
            'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR', 'AS', 'AT',
            'AU', 'AW', 'AX', 'AZ', 'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI',
            'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ', 'BR', 'BS', 'BT', 'BV', 'BW', 'BY',
            'BZ', 'CA', 'CC', 'CD', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN',
            'CO', 'CR', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM',
            'DO', 'DZ', 'EC', 'EE', 'EG', 'EH', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FK',
            'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GL',
            'GM', 'GN', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GW', 'GY', 'HK', 'HM',
            'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR',
            'IS', 'IT', 'JE', 'JM', 'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN',
            'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LI', 'LK', 'LR', 'LS',
            'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MK',
            'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW',
            'MX', 'MY', 'MZ', 'NA', 'NC', 'NE', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP',
            'NR', 'NU', 'NZ', 'OM', 'PA', 'PE', 'PF', 'PG', 'PH', 'PK', 'PL', 'PM',
            'PN', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS', 'RU', 'RW',
            'SA', 'SB', 'SC', 'SD', 'SE', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM',
            'SN', 'SO', 'SR', 'SS', 'ST', 'SV', 'SX', 'SY', 'SZ', 'TC', 'TD', 'TF',
            'TG', 'TH', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO', 'TR', 'TT', 'TV', 'TW',
            'TZ', 'UA', 'UG', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI',
            'VN', 'VU', 'WF', 'WS', 'YE', 'YT', 'ZA', 'ZM', 'ZW'
        ]

        if country_code not in valid_countries:
            return False

        # Luhn algorithm checksum validation
        try:
            return self._validate_isin_checksum(isin)
        except:
            return False

    def _validate_isin_checksum(self, isin: str) -> bool:
        """
        Validate ISIN using Luhn algorithm
        """
        # Convert letters to numbers (A=10, B=11, ..., Z=35)
        converted = ""
        for char in isin[:-1]:  # Exclude check digit
            if char.isalpha():
                converted += str(ord(char) - ord('A') + 10)
            else:
                converted += char

        # Apply Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(converted)):
            n = int(digit)
            if i % 2 == 1:  # Every second digit from right
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n

        check_digit = (10 - (total % 10)) % 10
        return check_digit == int(isin[-1])

    def validate_cusip(self, cusip: str) -> bool:
        """
        Validate CUSIP format
        """
        if not cusip or len(cusip) != 9:
            return False

        return bool(re.match(r'^[0-9A-Z]{9}$', cusip))

    def validate_sedol(self, sedol: str) -> bool:
        """
        Validate SEDOL format
        """
        if not sedol or len(sedol) != 7:
            return False

        return bool(re.match(r'^[0-9A-Z]{7}$', sedol))

    def extract_isin_from_text(self, text: str) -> Optional[str]:
        """
        Extract ISIN from text using regex
        """
        matches = re.findall(self.patterns['ISIN'], text.upper())
        for match in matches:
            if self.validate_isin(match):
                return match
        return None

    def extract_cusip_from_text(self, text: str) -> Optional[str]:
        """
        Extract CUSIP from text using regex
        """
        matches = re.findall(self.patterns['CUSIP'], text.upper())
        for match in matches:
            if self.validate_cusip(match):
                return match
        return None

    def extract_currency(self, text: str) -> Optional[str]:
        """
        Extract currency from text
        """
        # Check for currency codes
        currency_match = re.search(self.patterns['currency'], text.upper())
        if currency_match:
            return currency_match.group()

        # Check for currency symbols
        for symbol, code in self.currency_symbols.items():
            if symbol in text:
                return code

        return None

    def extract_amount(self, text: str) -> Optional[float]:
        """
        Extract monetary amount from text
        """
        # Remove currency symbols and clean text
        cleaned = text
        for symbol in self.currency_symbols.keys():
            cleaned = cleaned.replace(symbol, '')

        # Remove common currency codes
        for code in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']:
            cleaned = cleaned.replace(code, '')

        # Extract numeric pattern
        amount_patterns = [
            r'[\d,]+\.?\d*',  # 1,000.50 or 1000
            r'\d+\.\d{2}',  # 1000.50
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?'  # 1,000.50
        ]

        for pattern in amount_patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                amount_str = matches[0]

                # Handle different decimal separators
                if ',' in amount_str and '.' in amount_str:
                    # Assume last separator is decimal
                    if amount_str.rfind(',') > amount_str.rfind('.'):
                        amount_str = amount_str.replace('.', '').replace(',', '.')
                    else:
                        amount_str = amount_str.replace(',', '')
                elif ',' in amount_str:
                    # Check if it's likely a decimal separator
                    parts = amount_str.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2:
                        amount_str = amount_str.replace(',', '.')
                    else:
                        amount_str = amount_str.replace(',', '')

                try:
                    return float(amount_str)
                except ValueError:
                    continue

        return None

    def extract_date_from_text(self, text: str) -> Optional[str]:
        """
        Extract date from text
        """
        for pattern in self.patterns['date_patterns']:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None

    def extract_trade_id(self, text: str) -> Optional[str]:
        """
        Extract trade ID from text
        """
        match = re.search(self.patterns['trade_id'], text.upper())
        return match.group() if match else None

    def extract_account_id(self, text: str) -> Optional[str]:
        """
        Extract account ID from text
        """
        match = re.search(self.patterns['account'], text.upper())
        return match.group() if match else None

    def validate_financial_data(self, field_name: str, field_value: str) -> Dict[str, Any]:
        """
        Validate financial data based on field type
        """
        field_name_upper = field_name.upper()
        result = {
            'is_valid': False,
            'message': '',
            'suggested_correction': None
        }

        if field_name_upper == 'ISIN':
            result['is_valid'] = self.validate_isin(field_value)
            if not result['is_valid']:
                result['message'] = 'Invalid ISIN format or checksum'
                # Try to extract ISIN from the value
                extracted = self.extract_isin_from_text(field_value)
                if extracted:
                    result['suggested_correction'] = extracted

        elif field_name_upper == 'CUSIP':
            result['is_valid'] = self.validate_cusip(field_value)
            if not result['is_valid']:
                result['message'] = 'Invalid CUSIP format'

        elif field_name_upper == 'SEDOL':
            result['is_valid'] = self.validate_sedol(field_value)
            if not result['is_valid']:
                result['message'] = 'Invalid SEDOL format'

        elif field_name_upper in ['AMOUNT', 'VALUE']:
            amount = self.extract_amount(field_value)
            result['is_valid'] = amount is not None
            if result['is_valid']:
                result['suggested_correction'] = amount
            else:
                result['message'] = 'Invalid amount format'

        elif field_name_upper == 'CURRENCY':
            currency = self.extract_currency(field_value)
            result['is_valid'] = currency is not None
            if result['is_valid']:
                result['suggested_correction'] = currency
            else:
                result['message'] = 'Invalid currency format'

        else:
            result['is_valid'] = True
            result['message'] = 'Field type not validated'

        return result
