<human>
git-vector-grep finds likes in your repo based on embedding rather than regular expressions. Use it by asking your coding agent to use it to for its search functionality.

Caveat emptor. The tool is vibe-coded. I wanted to experiment with whether this helps. To be honest—I have no idea. Most models and harnesses have explore tools that work just fine with expanding grep.

The tool runs locally (does not call out to external embedding APIs) and stores state in git. You can share the state with others by pushing the relevant git refs and fetching them.

Installation and usage:

```sh
cargo install --git https://github.com/philz/git-vector-grep --locked git-vector-grep
git-vector-grep "where is authentication handled?"
```

The first search downloads a local model and indexes the repository. To share the index:

```sh
git-vector-grep push --remote origin
git-vector-grep pull --remote origin
```

License: MIT

Similar tools: git-semantic
</human>
