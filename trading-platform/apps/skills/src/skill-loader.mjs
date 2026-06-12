import { readFile, readdir, stat } from 'node:fs/promises';
import { join, basename, dirname } from 'node:path';

/**
 * Parse YAML frontmatter from a SKILL.md string.
 * Returns { frontmatter: Record<string,any>, body: string }.
 * Handles the common subset: scalar strings, nested keys one level deep.
 */
export function parseFrontmatter(raw) {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) {
    return { frontmatter: {}, body: raw.trim() };
  }

  const yamlBlock = match[1];
  const body = match[2].trim();
  const frontmatter = {};
  let currentKey = null;

  for (const line of yamlBlock.split('\n')) {
    const trimmed = line.trimEnd();
    if (trimmed === '') continue;

    const indented = line.match(/^(\s+)(\S.*)$/);
    if (indented && currentKey) {
      const nested = parseYamlLine(indented[2]);
      if (nested) {
        if (typeof frontmatter[currentKey] !== 'object' || frontmatter[currentKey] === null) {
          frontmatter[currentKey] = {};
        }
        frontmatter[currentKey][nested.key] = nested.value;
      }
      continue;
    }

    const parsed = parseYamlLine(trimmed);
    if (parsed) {
      frontmatter[parsed.key] = parsed.value;
      currentKey = parsed.key;
    }
  }

  return { frontmatter, body };
}

function parseYamlLine(line) {
  const m = line.match(/^([A-Za-z_][\w-]*)\s*:\s*(.*)$/);
  if (!m) return null;
  const key = m[1];
  let value = m[2].trim();
  if ((value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }
  if (value === '') value = null;
  return { key, value };
}

/**
 * Load a single SKILL.md file into a Skill object.
 * @param {string} filePath — absolute path to SKILL.md
 * @returns {Promise<import('./types.mjs').Skill>}
 */
export async function loadSkillFile(filePath) {
  const raw = await readFile(filePath, 'utf-8');
  const { frontmatter, body } = parseFrontmatter(raw);
  const dir = dirname(filePath);

  return {
    name: frontmatter.name ?? basename(dir),
    description: frontmatter.description ?? '',
    metadata: typeof frontmatter.metadata === 'object' ? frontmatter.metadata : {},
    body,
    filePath,
    directory: dir,
  };
}

/**
 * Recursively scan a directory for SKILL.md files and load them.
 * Each subdirectory with a SKILL.md is treated as one skill.
 * @param {string} rootDir — directory to scan
 * @returns {Promise<import('./types.mjs').Skill[]>}
 */
export async function loadSkillsFromDirectory(rootDir) {
  const skills = [];
  const entries = await readdir(rootDir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = join(rootDir, entry.name);

    if (entry.isDirectory()) {
      const skillFile = join(fullPath, 'SKILL.md');
      try {
        const s = await stat(skillFile);
        if (s.isFile()) {
          skills.push(await loadSkillFile(skillFile));
        }
      } catch {
        const nested = await loadSkillsFromDirectory(fullPath);
        skills.push(...nested);
      }
    }
  }

  return skills;
}
