-- Page-break control for the PDF: keep every table, every figure, and the headings directly
-- above them on one page. Before such a group, LaTeX's \needspace starts a new page unless the
-- group's estimated height still fits.

local TEXT_WIDTH_CM = 17.0 -- letter paper with 2.3 cm margins
local LINES_PER_CM = 2.09 -- 11 pt text
local CAPTION_LINES = 4

local function png_aspect(path)
  local file = io.open(path, "rb")
  if not file then
    return 0.75
  end
  local header = file:read(24)
  file:close()
  local width, height = string.unpack(">I4>I4", header, 17)
  return height / width
end

local function lines_needed(block)
  if block.t == "Table" then
    local rows = #block.head.rows
    for _, body in ipairs(block.bodies) do
      rows = rows + #body.body
    end
    return math.ceil(rows * 1.3) + CAPTION_LINES + 4
  elseif block.t == "Figure" then
    local image = pandoc.utils.blocks_to_inlines(block.content)[1]
    local width = tonumber((image.attributes.width or "100%"):match("[%d.]+")) / 100
    local height_cm = png_aspect("../" .. image.src) * width * TEXT_WIDTH_CM
    return math.ceil(height_cm * LINES_PER_CM) + CAPTION_LINES
  end
end

-- Height of block i plus any headings stacked directly on top of it.
local function group_lines(blocks, i)
  local block = blocks[i]
  if block.t == "Header" and blocks[i + 1] then
    local below = group_lines(blocks, i + 1)
    return below and below + 3
  end
  return lines_needed(block)
end

function Blocks(blocks)
  local out = pandoc.Blocks({})
  for i, block in ipairs(blocks) do
    local inside_group = i > 1 and blocks[i - 1].t == "Header"
    local lines = not inside_group and group_lines(blocks, i)
    if lines then
      out:insert(pandoc.RawBlock("latex", string.format("\\needspace{%d\\baselineskip}", lines)))
    end
    out:insert(block)
  end
  return out
end
