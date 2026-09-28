// OpenCode plugin (loaded by path from opencode.json, see core/agent.py
// opencode_config): a read with no line range on a long file gets the first
// READ_CAP lines and a note to grep for the function and read around it.
// The local model reads whole 3-7k-line files otherwise, and every later
// step waits while the LLM PC processes them again.
import fs from "fs"

const READ_CAP = 400

export const ReadCap = async () => {
  const capped = new Map()   // callID -> the file's line count
  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "read" || !output.args || output.args.limit) return
      let lines = 0
      try {
        const text = fs.readFileSync(output.args.filePath, "utf8")
        lines = text.replace(/\n$/, "").split("\n").length
      } catch (e) {
        return
      }
      if (lines <= READ_CAP) return
      output.args.limit = READ_CAP
      capped.set(input.callID, lines)
    },
    "tool.execute.after": async (input, output) => {
      if (!capped.has(input.callID)) return
      const lines = capped.get(input.callID)
      capped.delete(input.callID)
      output.output += `\n\n[Studio Assist: this file has ${lines} lines; only the first ` +
        `${READ_CAP} are shown. Do not page through it. Call repo_map on this file ` +
        `for its outline with line ranges, or repo_find with a name, then read with ` +
        `offset and limit.]`
    },
  }
}
