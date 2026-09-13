// Stage 5 task sourcing, option (d) from the review: stratify craft tasks by recipe-DAG depth instead of sampling
// uniformly. The recipe table is a DAG (result_item -> ingredients); depth
// is a free, principled difficulty axis -- depth 0 items need no crafting
// (raw materials), depth 1 items craft directly from depth-0 materials,
// depth N items need at least one depth-(N-1) ingredient first. Building the
// gradient in from the start means difficulty can be measured, not assumed.
//
// Scope: TIER-1-ONLY generation can't mine, so only items reachable through
// a chain of pure CRAFTING (never touching mine/place) starting from a small
// set of harness-supplied raw materials are usable -- in practice this means
// the wood-tool branch (logs -> planks -> sticks -> wooden tools/furniture),
// not the stone/iron/diamond branches (those need mine() to gather ore,
// which is tier-2 and out of scope until placeBlock/attack are resolved and
// the reviewer's ungated-generation-server design for tier 2/3 is built). This is a
// REAL scoping limit, surfaced explicitly via reachability stats below, not
// hidden.
// KNOWN LIMITATION, inherited from the oracle DB's own recipe schema, not
// introduced here: a result_item's ingredient list flattens ALTERNATE
// recipes together with true compound ingredients, with no way to tell them
// apart from the row data alone. Confirmed live: torch's ingredients are
// ["charcoal","coal","stick"] -- really "(charcoal OR coal) AND stick" in
// real Minecraft, but the schema (and gen_query_traces.py's existing
// "X, Y, or Z" English rendering, a pre-existing project convention, not
// something added for this) can't distinguish that from "needs charcoal AND
// coal AND stick simultaneously." This module treats every listed
// ingredient as REQUIRED (AND) -- the conservative choice: it may
// under-count reachable items (e.g. 'stick' looks unreachable here without
// bamboo even though planks alone would satisfy the real alternate recipe),
// but it never claims an item reachable when the DB can't actually back
// that up. Task SELECTION only loses some valid candidates this way; it
// never generates a task for something that isn't truly craftable.
async function fetchAllRecipes(knowledgeServerBase) {
  const res = await fetch(`${knowledgeServerBase}/all_recipes`)
  if (!res.ok) throw new Error(`all_recipes HTTP ${res.status}`)
  const { recipes } = await res.json()
  // Drop tag-reference "recipes" (result_item starting with '#') -- those
  // are trim/template pattern rows, not concrete craftable items.
  const clean = {}
  for (const [item, ingredients] of Object.entries(recipes)) {
    if (item.startsWith('#')) continue
    clean[item] = [...new Set(ingredients)] // de-dup repeated grid-slot ingredients
  }
  return clean
}

// Tag ingredients in recipe rows are '#'-prefixed (e.g. '#minecraft:planks')
// but the DB's own tag table stores names WITHOUT the '#' (confirmed live:
// /tag_members?tag=%23minecraft:oak_logs returns [] while
// /tag_members?tag=minecraft:oak_logs returns the real 4 members) -- strip
// it before querying.
async function fetchTagMembers(knowledgeServerBase, tag, cache) {
  const bare = tag.replace(/^#/, '')
  if (cache.has(bare)) return cache.get(bare)
  const res = await fetch(`${knowledgeServerBase}/tag_members?tag=${encodeURIComponent(bare)}`)
  if (!res.ok) throw new Error(`tag_members HTTP ${res.status} for ${bare}`)
  const { members } = await res.json()
  cache.set(bare, members)
  return members
}

// Pre-resolves every distinct tag ingredient appearing in the recipe set
// into its concrete member list, once, so computeDepths/reachableTier1 can
// treat "ingredient is tag T" as "ingredient is ANY of T's real members"
// without doing network calls mid-recursion.
async function resolveAllTags(knowledgeServerBase, recipes) {
  const tags = new Set()
  for (const ings of Object.values(recipes)) {
    for (const ing of ings) if (ing.startsWith('#')) tags.add(ing)
  }
  const cache = new Map()
  for (const tag of tags) await fetchTagMembers(knowledgeServerBase, tag, cache)
  return cache // bare-tag-name -> [member, ...]
}

// depth(item): 0 for a raw material (no recipe, or every ingredient is
// itself unreachable/opaque -- e.g. a tag reference '#minecraft:...' with no
// bulk membership data here, treated as opaque rather than chased further).
// Memoized with cycle protection (recipe graphs shouldn't cycle, but a
// defensive guard costs nothing).
function computeDepths(recipes, tagMembers = new Map()) {
  const depth = new Map()
  const inProgress = new Set()
  function depthOf(item) {
    if (depth.has(item)) return depth.get(item)
    const ingredients = recipes[item]
    if (!ingredients || ingredients.length === 0) {
      depth.set(item, 0)
      return 0
    }
    if (inProgress.has(item)) {
      // cycle -- shouldn't happen in a real recipe graph; treat as opaque
      // rather than infinite-recursing.
      depth.set(item, 0)
      return 0
    }
    inProgress.add(item)
    let maxIngredientDepth = 0
    for (const ing of ingredients) {
      if (ing.startsWith('#')) {
        // A tag ingredient is satisfied by whichever real member is
        // shallowest to craft -- use the MIN over members as that
        // ingredient's effective depth.
        const members = tagMembers.get(ing.replace(/^#/, '')) || []
        if (members.length === 0) continue // unresolved tag, treat as opaque/depth-0
        const minMemberDepth = Math.min(...members.map((m) => depthOf(m)))
        maxIngredientDepth = Math.max(maxIngredientDepth, minMemberDepth)
        continue
      }
      maxIngredientDepth = Math.max(maxIngredientDepth, depthOf(ing))
    }
    inProgress.delete(item)
    const d = 1 + maxIngredientDepth
    depth.set(item, d)
    return d
  }
  for (const item of Object.keys(recipes)) depthOf(item)
  return depth
}

// reachableTier1(recipes, depths, rawMaterials): which items can be crafted
// using ONLY tier-1 tools (craft, no mine/place) starting from a given set
// of raw materials the harness can supply directly (e.g. oak_log). An item
// is reachable if every one of its ingredients is either a supplied raw
// material or itself reachable (recursively) -- i.e. the whole ingredient
// chain bottoms out in raw materials we actually have, never in something
// that would need mining.
function reachableTier1(recipes, rawMaterials, tagMembers = new Map()) {
  const raw = new Set(rawMaterials)
  const reachable = new Map() // item -> ordered list of craft steps to reach it (each step = item to craft)
  const inProgress = new Set()
  let calls = 0
  const MAX_CALLS = 200000 // circuit breaker -- if this trips, something's exponential, not just slow

  // For a tag ingredient, pick the SHALLOWEST reachable member (fewest
  // prerequisite steps) -- e.g. '#minecraft:planks' resolves to whichever
  // plank color is actually reachable from the supplied raw materials.
  // MEMOIZED per tag (tagResolutionCache) -- without this, every item that
  // depends on a common tag (e.g. many colored-bundle items all needing
  // '#minecraft:wool') re-walks all of that tag's members from scratch on
  // every occurrence, which blew up exponentially on the real 1008-recipe
  // graph (confirmed live: tripped a 200k-call circuit breaker before this
  // fix). resolve()'s own per-item memoization couldn't save this because
  // resolveTag's iteration overhead across members was what repeated, not
  // any single item's resolution.
  const tagResolutionCache = new Map()
  function resolveTag(tag) {
    const bare = tag.replace(/^#/, '')
    if (tagResolutionCache.has(bare)) return tagResolutionCache.get(bare)
    const members = tagMembers.get(bare) || []
    let best = null
    for (const m of members) {
      if (raw.has(m)) { best = { item: m, steps: [] }; break }
      const s = resolve(m)
      if (s !== null && (best === null || s.length < best.steps.length)) best = { item: m, steps: s }
    }
    tagResolutionCache.set(bare, best)
    return best
  }

  function resolve(item) {
    calls++
    if (calls > MAX_CALLS) throw new Error(`reachableTier1 exceeded ${MAX_CALLS} calls -- likely exponential blowup, not just slow (last item: ${item})`)
    if (raw.has(item)) return []
    if (reachable.has(item)) return reachable.get(item)
    if (inProgress.has(item)) return null
    const ingredients = recipes[item]
    if (!ingredients) return null // no recipe and not raw -- unreachable (needs mining/other source)
    inProgress.add(item)
    const steps = []
    for (const ing of ingredients) {
      if (ing.startsWith('#')) {
        const resolved = resolveTag(ing)
        if (resolved === null) { inProgress.delete(item); return null }
        for (const s of resolved.steps) if (!steps.includes(s)) steps.push(s)
        if (!raw.has(resolved.item) && !steps.includes(resolved.item)) steps.push(resolved.item)
        continue
      }
      if (raw.has(ing)) continue
      const ingSteps = resolve(ing)
      if (ingSteps === null) { inProgress.delete(item); return null }
      for (const s of ingSteps) if (!steps.includes(s)) steps.push(s)
      if (!steps.includes(ing)) steps.push(ing)
    }
    inProgress.delete(item)
    reachable.set(item, steps)
    return steps
  }
  for (const item of Object.keys(recipes)) resolve(item)
  return reachable // item -> [prerequisite items to craft, in order, NOT including item itself]
}

module.exports = { fetchAllRecipes, resolveAllTags, computeDepths, reachableTier1 }
