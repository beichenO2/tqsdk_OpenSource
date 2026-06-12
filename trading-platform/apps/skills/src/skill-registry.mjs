import { loadSkillsFromDirectory } from './skill-loader.mjs';

/**
 * In-memory registry of available skills.
 * Supports register, unregister, lookup, and bulk loading from directories.
 */
export class SkillRegistry {
  /** @type {Map<string, import('./types.mjs').Skill>} */
  #skills = new Map();

  /** Register a single skill. Overwrites if name already exists. */
  register(skill) {
    if (!skill?.name) {
      throw new Error('Skill must have a name');
    }
    this.#skills.set(skill.name, skill);
    return this;
  }

  /** Unregister a skill by name. Returns true if it existed. */
  unregister(name) {
    return this.#skills.delete(name);
  }

  /** Get a skill by name, or undefined. */
  get(name) {
    return this.#skills.get(name);
  }

  /** Check if a skill is registered. */
  has(name) {
    return this.#skills.has(name);
  }

  /** Return all registered skills as an array. */
  list() {
    return Array.from(this.#skills.values());
  }

  /** Return all registered skill names. */
  names() {
    return Array.from(this.#skills.keys());
  }

  /** Number of registered skills. */
  get size() {
    return this.#skills.size;
  }

  /**
   * Scan a directory for SKILL.md files and register all found skills.
   * @param {string} directory — root directory to scan
   * @returns {Promise<string[]>} — names of newly registered skills
   */
  async loadDirectory(directory) {
    const skills = await loadSkillsFromDirectory(directory);
    const names = [];
    for (const skill of skills) {
      this.register(skill);
      names.push(skill.name);
    }
    return names;
  }

  /** Remove all registered skills. */
  clear() {
    this.#skills.clear();
    return this;
  }

  /**
   * Find skills matching a query string (searches name + description).
   * @param {string} query
   * @returns {import('./types.mjs').Skill[]}
   */
  search(query) {
    const q = query.toLowerCase();
    return this.list().filter(
      (s) => s.name.toLowerCase().includes(q) || s.description.toLowerCase().includes(q),
    );
  }
}
