#!/usr/bin/env node
/**
 * Reads real backtest/optimizer JSON files from the project
 * and generates a consolidated static data file for the web frontend.
 *
 * Run: node scripts/generate-static-data.mjs
 * Output: public/data/backtests.json, public/data/optimizer.json
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'fs';
import { join, basename } from 'path';

const MODELS_DIR = join(import.meta.dirname, '../../../models');
const OPTIMIZER_DIR = join(import.meta.dirname, '../../../eternal-optimizer/results');
const OUT_DIR = join(import.meta.dirname, '../public/data');

mkdirSync(OUT_DIR, { recursive: true });

// --- Backtest results ---
function loadBacktests() {
  const results = [];
  const files = readdirSync(MODELS_DIR).filter(f => f.startsWith('backtest_') && f.endsWith('.json'));

  for (const file of files) {
    try {
      const raw = JSON.parse(readFileSync(join(MODELS_DIR, file), 'utf-8'));
      const name = basename(file, '.json').replace('backtest_', '');

      if (typeof raw === 'object' && !Array.isArray(raw)) {
        for (const [strategyKey, metrics] of Object.entries(raw)) {
          if (typeof metrics !== 'object' || metrics === null) continue;
          const m = metrics;

          // Determine market type
          const isCrypto = name.includes('USDT') || name.includes('BTC') || name.includes('ETH') || name.includes('SOL') || name.includes('BNB') || name.includes('XRP');
          const isSplit = name.includes('_split');

          results.push({
            id: `${name}_${strategyKey}`,
            source_file: file,
            symbol: name.replace('_split', ''),
            strategy_name: strategyKey,
            market: isCrypto ? 'crypto' : 'futures',
            is_split: isSplit,
            split_type: isSplit ? (strategyKey === 'train' || strategyKey === 'is' ? strategyKey : null) : null,
            initial_capital: m.initial_capital ?? 100000,
            final_capital: m.final_capital ?? m.initial_capital ?? 100000,
            total_return: m.total_return ?? 0,
            max_drawdown: m.max_drawdown ?? 0,
            sharpe_ratio: m.sharpe_ratio ?? m.sharpe ?? 0,
            calmar_ratio: m.calmar_ratio ?? null,
            total_trades: m.total_trades ?? 0,
            win_rate: m.win_rate ?? 0,
            profit_factor: m.profit_factor === 'inf' ? 999 : (m.profit_factor ?? 0),
            avg_win: m.avg_win ?? 0,
            avg_loss: m.avg_loss ?? 0,
            payoff_ratio: m.payoff_ratio ?? 0,
            total_commission: m.total_commission ?? 0,
            bars_tested: m.bars_tested ?? 0,
            date_range: m.date_range ?? '',
            round_trips: m.round_trips ?? 0,
          });
        }
      }
    } catch (e) {
      console.warn(`Skip ${file}: ${e.message}`);
    }
  }

  // Also load optuna results
  const optunaFiles = ['optuna_v4_blend.json', 'optuna_results.json', 'optuna_sota_results.json', 'optuna_crypto_results.json'];
  const optuna = [];
  for (const file of optunaFiles) {
    try {
      const raw = JSON.parse(readFileSync(join(MODELS_DIR, file), 'utf-8'));
      optuna.push({ file, data: raw });
    } catch { /* skip */ }
  }

  return { backtests: results, optuna };
}

// --- Optimizer convergence ---
function loadOptimizerResults() {
  let files;
  try {
    files = readdirSync(OPTIMIZER_DIR).filter(f => f.startsWith('round_') && f.endsWith('.json')).sort();
  } catch {
    return [];
  }

  const rounds = [];
  // Sample: take every 10th round to keep JSON size manageable
  const step = Math.max(1, Math.floor(files.length / 100));

  for (let i = 0; i < files.length; i += step) {
    try {
      const raw = JSON.parse(readFileSync(join(OPTIMIZER_DIR, files[i]), 'utf-8'));
      const roundNum = parseInt(files[i].match(/round_(\d+)/)?.[1] ?? '0');
      rounds.push({
        round: roundNum,
        file: files[i],
        best_score: raw.best_score ?? raw.score ?? null,
        sharpe: raw.sharpe ?? raw.best_sharpe ?? null,
        total_return: raw.total_return ?? null,
        max_drawdown: raw.max_drawdown ?? null,
        params: raw.best_params ?? raw.params ?? null,
      });
    } catch { /* skip */ }
  }

  return rounds;
}

// --- Generate ---
const { backtests, optuna } = loadBacktests();
const optimizer = loadOptimizerResults();

writeFileSync(join(OUT_DIR, 'backtests.json'), JSON.stringify({ backtests, optuna }, null, 0));
writeFileSync(join(OUT_DIR, 'optimizer.json'), JSON.stringify(optimizer, null, 0));

console.log(`Generated:`);
console.log(`  backtests.json: ${backtests.length} backtest results, ${optuna.length} optuna files`);
console.log(`  optimizer.json: ${optimizer.length} rounds (sampled from ${readdirSync(OPTIMIZER_DIR).length})`);
