import sys
import io
import json


def make_tree(spec, cls):
    if spec is None:
        return None
    val, left, right = spec
    return cls(val, make_tree(left, cls), make_tree(right, cls))


def main():
    with open(sys.argv[1]) as f:
        payload = json.load(f)

    code = payload["code"]
    func_name = payload["func_name"]
    test_cases = payload["test_cases"]

    results = {"define_error": None, "tests": []}

    # supplied in case the participant's code assumes TreeNode is already
    # defined (as the symmetric-tree question's prompt describes it) rather
    # than defining it themselves
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    namespace = {"TreeNode": TreeNode}

    try:
        exec(code, namespace)
    except Exception as e:
        results["define_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(results))
        return

    tree_cls = namespace.get("TreeNode", TreeNode)
    func = namespace.get(func_name)

    if not callable(func):
        results["define_error"] = f"function '{func_name}' not found"
        print(json.dumps(results))
        return

    for tc in test_cases:
        entry = {"expected": tc["expected"]}
        try:
            if tc["kind"] == "args":
                entry["actual"] = func(*tc["args"])
                entry["error"] = None
            elif tc["kind"] == "tree":
                root = make_tree(tc["tree"], tree_cls)
                entry["actual"] = func(root)
                entry["error"] = None
            elif tc["kind"] == "stdout":
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    func(*tc["args"])
                finally:
                    sys.stdout = old_stdout
                entry["actual"] = sorted({line.strip() for line in buf.getvalue().splitlines() if line.strip()})
                entry["error"] = None
            else:
                entry["actual"] = None
                entry["error"] = f"unknown test kind '{tc['kind']}'"
        except Exception as e:
            entry["actual"] = None
            entry["error"] = f"{type(e).__name__}: {e}"
        results["tests"].append(entry)

    try:
        output = json.dumps(results)
    except TypeError:
        for entry in results["tests"]:
            try:
                json.dumps(entry["actual"])
            except TypeError:
                entry["actual"] = str(entry["actual"])
        output = json.dumps(results)

    print(output)


if __name__ == "__main__":
    main()
