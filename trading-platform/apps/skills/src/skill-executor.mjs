import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

/**
 * @typedef {Object} ExecutionContext
 * @property {Record<string, string>} [variables] — template variables to interpolate
 * @property {string} [workingDirectory] — cwd for any shell steps
 * @property {boolean} [dryRun] — if true, return instructions without executing shell commands
 */

/**
 * @typedef {Object} ExecutionResult
 * @property {string} skillName
 * @property {string} instructions — resolved skill body (with variables interpolated)
 * @property {string[]} shellCommands — extracted shell commands from code blocks
 * @property {boolean} dryRun
 */

/**
 * Extract fenced shell/bash code blocks from markdown body.
 * @param {string} body
 * @returns {string[]}
 */
export function extractShellCommands(body) {
  const commands = [];
  const regex = /```(?:sh|bash|shell)\n([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(body)) !== null) {
    const cmd = match[1].trim();
    if (cmd) commands.push(cmd);
  }
  return commands;
}

/**
 * Replace {{VARIABLE}} placeholders in text with provided values.
 * @param {string} text
 * @param {Record<string, string>} variables
 * @returns {string}
 */
export function interpolateVariables(text, variables) {
  if (!variables || Object.keys(variables).length === 0) return text;
  return text.replace(/\{\{(\w+)\}\}/g, (full, key) => {
    return key in variables ? variables[key] : full;
  });
}

/**
 * Execute (resolve) a loaded skill.
 *
 * "Execution" means:
 * 1. Interpolate template variables into the skill body
 * 2. Extract shell commands from fenced code blocks
 * 3. Return the resolved instructions (and commands if any)
 *
 * This does NOT spawn child processes — the caller decides how to run commands.
 *
 * @param {import('./types.mjs').Skill} skill
 * @param {ExecutionContext} [context]
 * @returns {Promise<ExecutionResult>}
 */
export async function executeSkill(skill, context = {}) {
  const { variables = {}, dryRun = true } = context;

  let body = skill.body;

  const filesBlock = body.match(/<files_to_read>([\s\S]*?)<\/files_to_read>/);
  const filePaths = [];
  if (filesBlock) {
    const paths = filesBlock[1].trim().split('\n').map((l) => l.trim()).filter(Boolean);
    for (const p of paths) {
      const resolved = p.startsWith('/') ? p : join(skill.directory, p);
      try {
        const content = await readFile(resolved, 'utf-8');
        filePaths.push({ path: resolved, content });
      } catch {
        filePaths.push({ path: resolved, content: null, error: 'file_not_found' });
      }
    }
  }

  const instructions = interpolateVariables(body, variables);
  const shellCommands = extractShellCommands(instructions);

  return {
    skillName: skill.name,
    instructions,
    shellCommands,
    dryRun,
    resolvedFiles: filePaths,
  };
}
