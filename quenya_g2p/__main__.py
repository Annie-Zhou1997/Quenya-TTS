"""Prints the IPA for Quenya text: python -m quenya_g2p "Elen síla lúmenn' omentielvo" (or text on stdin)."""

import sys

from quenya_g2p import to_ipa


def main():
    lines = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else sys.stdin.read().splitlines()
    for line in lines:
        if line.strip():
            print(to_ipa(line))


if __name__ == "__main__":
    main()
