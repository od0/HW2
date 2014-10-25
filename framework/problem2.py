import decision_tree as dt
import scan
import utils

INPUT_FILE = ('/Users/sean/Sync/cornell/modern-analytics-fall-2014/'
  'hw2-bucket/data/finefoods.txt')

def main():
    binary_label = True

    data = scan.scan(INPUT_FILE, binary_label)
    length = len(data)

    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]

    decision_tree = dt.train(train_data)
    test_results = dt.test(decision_tree, test_data)

    print test_results

if __name__ == '__main__':
    main()
