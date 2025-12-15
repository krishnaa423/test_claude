"""
Command-line interface for DFT plane wave code.
"""

import argparse
import sys


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='DFT Plane Wave Calculator'
    )
    parser.add_argument(
        '--version', action='version', version='dft_pw 0.1.0'
    )

    subparsers = parser.add_subparsers(dest='command')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run a test calculation')
    test_parser.add_argument(
        '--system', default='Si', help='System to test (Si)'
    )

    args = parser.parse_args()

    if args.command == 'test':
        from .calculator import silicon_test
        silicon_test()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
