import os
import pickle
import shutil


class PickleList:
    """
    The purpose of this class is to define a data structure that has an
    interface very similar to that of a Python list. However, instead of
    storing its items in RAM as in a normal python list, a pickle list
    stores each of its items as separate pickle files. This is useful when
    the items are too big to store them all at once in RAM.

    """

    def __init__(self, base_dir):
        """
        Constructor

        Parameters
        ----------
        base_dir: str
            all items in the pickle list will be stored in a folder with
            this name, and named "item0", "item1", "item2", ..., etc.
        """
        self.base_dir = base_dir
        self.index = 0
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)

    def restart(self):
        """
        This method empties the base directory, but keeps the directory itself.

        Returns
        -------
        None

        """
        shutil.rmtree(self.base_dir)
        os.makedirs(self.base_dir)
        self.index = 0

    def __len__(self):
        """
        This method gives the number of items in the list, which is the same
        as the number of pickle files in the base directory.

        Returns
        -------
        int

        """
        return self.index

    def __getitem__(self, index):
        """
        This method allows the user to access an item in a pickle list li by
        using li[index] where index is a 0-based int within the range 0:len(
        li).

        Parameters
        ----------
        index: int

        Returns
        -------
        Any

        """
        item_name = f'item{index}'
        file_path = os.path.join(self.base_dir, item_name)

        if not os.path.exists(file_path):
            raise IndexError(f"Index {index} out of range")

        with open(file_path, 'rb') as file:
            item = pickle.load(file)

        return item

    def append(self, item):
        """
        This method allows one to append items to the pickle list the same
        way one does with a python list.

        Parameters
        ----------
        item: Any

        Returns
        -------

        """
        item_name = f'item{self.index}'
        file_path = os.path.join(self.base_dir, item_name)
        with open(file_path, 'wb') as file:
            pickle.dump(item, file)
        self.index += 1

    def __iter__(self):
        """
        This method returns a ListIterator (i.e., an iterator for the pickle
        list). This in turn allows statements like `for x in li:`,
        where `li` is a pickle list.

        Returns
        -------
        ListIterator

        """
        return self.ListIterator(self.base_dir)

    class ListIterator:
        """
        This class creates an iterator for a pickle list. Objects of this
        class should only be used internally by class PickleList. That is
        why this class is declared inside class PickleList.

        """

        def __init__(self, base_dir):
            """
            Constructor

            Parameters
            ----------
            base_dir: str
            """
            self.base_dir = base_dir
            self.index = 0

        def __iter__(self):
            """
            This method returns self.

            Returns
            -------
            ListIterator

            """
            return self

        def __next__(self):
            """
            This method returns the next item in the pickle list. That item
            is obtained by reading it from a pickle file.

            Returns
            -------
            Any

            """
            item_name = f'item{self.index}'
            file_path = os.path.join(self.base_dir, item_name)

            if not os.path.exists(file_path):
                raise StopIteration

            with open(file_path, 'rb') as file:
                item = pickle.load(file)

            # Delete the file after loading it
            # os.remove(file_path)

            self.index += 1
            return item


if __name__ == "__main__":
    def main():
        base_dir = "example_PickleList_files"
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
