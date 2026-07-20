import ast


def extract_helper_definitions(
    reference_code: str,
    target_function_name: str,
) -> str:
    """
    Extract helper classes/functions from the reference solution,
    excluding the target function itself.
    """

    tree = ast.parse(reference_code)

    helper_nodes = []

    for node in tree.body:

        if isinstance(node, ast.ClassDef):
            helper_nodes.append(node)

        elif isinstance(node, ast.FunctionDef):
            if node.name != target_function_name:
                helper_nodes.append(node)

    helper_module = ast.Module(
        body=helper_nodes,
        type_ignores=[]
    )

    return ast.unparse(helper_module)