local mp = require("mp")
local msg = require("mp.msg")
local options = require("mp.options")

local opts = {
    csv_path = "audio_click_annotations.csv",
    show_osd = "yes",
    dedupe = "yes",
}

options.read_options(opts, "audio_click_logger")

local seen_paths = {}

local function csv_escape(value)
    local s = tostring(value or "")
    s = s:gsub('"', '""')
    return '"' .. s .. '"'
end

local function basename(path)
    if not path then
        return ""
    end
    return path:match("([^/\\]+)$") or path
end

local function ensure_header()
    local has_content = false
    local existing = io.open(opts.csv_path, "r")
    if existing then
        has_content = existing:read(1) ~= nil
        existing:close()
    end
    if has_content then
        return true
    end

    local f, err = io.open(opts.csv_path, "a")
    if not f then
        msg.error("unable to open CSV for header: " .. tostring(err))
        return false
    end
    f:write("file_name,file_path\n")
    f:close()
    return true
end

local function append_click_row(path)
    if not ensure_header() then
        if opts.show_osd == "yes" then
            mp.osd_message("CSV unavailable", 1.2)
        end
        return false
    end

    local row = table.concat({
        csv_escape(basename(path)),
        csv_escape(path),
    }, ",") .. "\n"

    local f, open_err = io.open(opts.csv_path, "a")
    if not f then
        msg.error("unable to append CSV: " .. tostring(open_err))
        if opts.show_osd == "yes" then
            mp.osd_message("CSV open error", 1.2)
        end
        return false
    end

    local ok, write_err = f:write(row)
    f:close()

    if not ok then
        msg.error("unable to write CSV row: " .. tostring(write_err))
        if opts.show_osd == "yes" then
            mp.osd_message("CSV write error", 1.2)
        end
        return false
    end

    if opts.show_osd == "yes" then
        mp.osd_message("Selected: " .. basename(path), 0.8)
    end
    return true
end

local function on_left_click(event)
    if event.event ~= "down" then
        return
    end

    local path = mp.get_property("path") or ""
    if path == "" then
        if opts.show_osd == "yes" then
            mp.osd_message("No active file", 0.8)
        end
        return
    end

    if opts.dedupe == "yes" and seen_paths[path] then
        if opts.show_osd == "yes" then
            mp.osd_message("Already selected: " .. basename(path), 0.8)
        end
        return
    end

    if append_click_row(path) then
        seen_paths[path] = true
    end
end

mp.register_event("file-loaded", ensure_header)
mp.add_forced_key_binding("MBTN_LEFT", "audio_click_logger_left_click", on_left_click, { complex = true })
msg.info("audio_click_logger loaded; CSV: " .. opts.csv_path)
