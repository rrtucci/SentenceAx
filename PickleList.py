import os
import pickle

class PickleList:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.index = 0
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def __len__(self):
        return self.index

    def __getitem__(self, index):
        item_name = f'item{index}'
        file_path = os.path.join(self.base_dir, item_name)

        if not os.path.exists(file_path):
            raise IndexError(f"Index {index} out of range")

        with open(file_path, 'rb') as file:
            item = pickle.load(file)

        return item

    def append(self, item):
        item_name = f'item{self.index}'
        file_path = os.path.join(self.base_dir, item_name)
        with open(file_path, 'wb') as file:
            pickle.dump(item, file)
        self.index += 1

    def __iter__(self):
        return self.ListIterator(self.base_dir)

    class ListIterator:
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            item_name = f'item{self.index}'
            file_path = os.path.join(self.base_dir, item_name)

            if not os.path.exists(file_path):
                raise StopIteration

            with open(file_path, 'rb') as file:
                item = pickle.load(file)

            # Delete the file after loading it
            os.remove(file_path)

            self.index += 1
            return item


if __name__ == "__main__":
    def main():
        base_dir = "pickle_files"
        plist = PickleList(base_dir)

        # Appending items to the list and storing in separate pickle files
        plist.append("Item 0")
        plist.append("Item 1")
        plist.append("Item 2")

        print(plist[1])
        print(len(plist))

        # Iterating over the stored items, deleting the pickle files as we go
        for item in plist:
            print(item)

    main()
