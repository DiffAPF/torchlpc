import re


def format_branch_name(name):
    # "(fix|feat)/issue-name" or CICD's branch "HEAD"
    pattern = re.compile("^((fix|feat)\/(?P<branch>.+))|((head|HEAD))")

    match = pattern.search(name)
    if match:
        return f"dev+{match.group(0)}"  # => dev+"(fix|feat)/issue-name"

    # function is called even if branch name is not used in a current template
    # just left properly named branches intact
    if name in ["master", "dev", "main"]:
        return name

    # fail in case of wrong branch names like "bugfix/issue-unknown"
    raise ValueError(f"Wrong branch name: {name}")
