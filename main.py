import sys


def main():
    argc = len(sys.argv)

    if sys.argv[1] == "train":
        # code for torch model training
        print("Training model...")
    elif sys.argv[1] == "test":
        # code for torch model testing
        print("Testing model...")
    else:
        print("Invalid argument")
        sys.exit(1)


if __name__ == "__main__":
    main()