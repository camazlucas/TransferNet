import argparse
import json

from .data import build_rel2id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', nargs='+', required=True, help='one or more CSVs with a "paths" column')
    parser.add_argument('--output', required=True, help='path to write the rel2id JSON')
    args = parser.parse_args()

    rel2id = build_rel2id(args.input_csv)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(rel2id, f, ensure_ascii=False, indent=2)

    print('{} relations (+ STOP) written to {}'.format(len(rel2id) - 1, args.output))


if __name__ == '__main__':
    main()
