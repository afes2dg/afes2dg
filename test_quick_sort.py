import unittest
import ast

file_path = "ㅎ3 DQZ _ cpu ram - nvidia gtx10 이상.py"

with open(file_path, "r", encoding="utf-8") as f:
    source_code = f.read()

# Parse AST and extract quick_sort function AST node
parsed_ast = ast.parse(source_code, filename=file_path)
quick_sort_node = None
for node in ast.walk(parsed_ast):
    if isinstance(node, ast.FunctionDef) and node.name == "quick_sort":
        quick_sort_node = node
        break

# Create module AST containing only quick_sort
module_ast = ast.Module(body=[quick_sort_node], type_ignores=[])
compiled_code = compile(module_ast, filename=file_path, mode="exec")

scope = {}
exec(compiled_code, scope)
quick_sort = scope["quick_sort"]


class TestQuickSort(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(quick_sort([]), [])

    def test_single_element(self):
        self.assertEqual(quick_sort([42]), [42])

    def test_sorted_list(self):
        self.assertEqual(quick_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_reverse_sorted_list(self):
        self.assertEqual(quick_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_unsorted_list(self):
        self.assertEqual(quick_sort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]), [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9])

    def test_duplicates(self):
        arr = [2, 2, 2, 1, 2, 3, 2]
        expected = sorted(arr)
        result = quick_sort(arr)
        self.assertEqual(len(result), len(expected))
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
