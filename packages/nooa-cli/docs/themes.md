# TUI themes

NOOA ships four themes: `mocha`, `latte`, `vsdark`, and `vslight`.

Use `/theme` with no arguments to open the full-screen theme browser. Moving through the list previews the complete theme immediately, including syntax-highlighted code and a unified diff. **Enter** applies and saves it, while **Esc** or **q** closes the browser and restores the theme that was active when it opened. `/theme <id>` remains the scriptable shortcut.

## Semantic color roles

Theme consumers should prefer these semantic roles instead of choosing a palette hue directly:

| Role | Used for |
|---|---|
| `text_primary`, `text_muted`, `text_subtle` | primary and secondary copy |
| `surface_raised`, `border_default` | panels and separators |
| `feedback_success`, `feedback_error`, `feedback_warning`, `feedback_info` | status feedback |
| `selection_fg`, `selection_bg` | selected text, rows, and focused controls |
| `search_match_fg`, `search_match_bg` | ordinary search matches |
| `search_current_fg`, `search_current_bg` | current search occurrence |
| `focus_accent` | active-pane rail |
| `user_message_fg`, `user_message_bg` | user-message bars |
| `inline_code_fg`, `inline_code_bg` | Markdown inline-code chips |
| `code_path`, `code_number` | technical values |
| `diff_added`, `diff_removed` | added and removed diff lines |

The legacy Catppuccin-named palette keys remain available as source swatches and for compatibility. New UI code should consume semantic roles. The Rich, prompt-toolkit, and ANSI adapters all resolve selection, search, focus, user-message, inline-code, and diff styles from these semantic roles.

Built-in and installed themes use the same catalog parser, semantic-role expansion, contrast validation, and rendering path. The built-ins are Base24 definitions with semantic overrides; downloaded Base16/Base24 schemes follow the identical loading path.

## Installing a theme

Place `.yaml` or `.yml` files in either directory:

- User-wide: `~/.config/nooa/themes/`
- Project-local: `<project>/.nooa/themes/`

Project themes override user themes with the same ID. Invalid files are skipped with a warning rather than preventing startup. Opening `/theme` reloads both directories, so adding a file does not require restarting the TUI.

To install a theme downloaded from the internet, choose a Base16/Base24 YAML scheme (for example, from the [Tinted Theming schemes collection](https://github.com/tinted-theming/schemes)), copy its **raw** file URL, and save it in the user directory:

```bash
mkdir -p ~/.config/nooa/themes
curl -L "$RAW_THEME_URL" -o ~/.config/nooa/themes/my-theme.yaml
```

Review downloaded files before using them. The filename becomes the theme ID when the file does not declare `slug` or `id`; open `/theme` to reload and preview it.

To create a theme, the simplest route is to copy the Base16 example below to `~/.config/nooa/themes/my-theme.yaml`, change its `scheme`, `slug`, and colors, then run `/theme`. Add optional semantic-role keys at the top level when you need exact control over selection, inline code, or diff colors.

### Base16 and Base24

Standard Base16 YAML files are accepted directly. A minimal example:

```yaml
scheme: Ocean
slug: ocean
variant: dark
base00: 2b303b
base01: 343d46
base02: 4f5b66
base03: 65737e
base04: a7adba
base05: c0c5ce
base06: dfe1e8
base07: eff1f5
base08: bf616a
base09: d08770
base0A: ebcb8b
base0B: a3be8c
base0C: 96b5b4
base0D: 8fa1b3
base0E: b48ead
base0F: ab7967
```

Base24 files are accepted when all extension keys `base10` through `base17` are present. NOOA maps Base16 swatches to semantic UI roles and rejects required text/highlight combinations below the configured contrast thresholds.

### Semantic overrides

Base16 and Base24 files may override any semantic role listed above at the top level. For example:

```yaml
# Include base00 through base0F as shown above, then optionally add:
inline_code_fg: '#ffffff'
inline_code_bg: '#005fb8'
selection_fg: '#ffffff'
selection_bg: '#264f78'
diff_added: '#287a1f'
diff_removed: '#b42318'
```

Only six-digit RGB values are accepted. `syntax_theme` must name an installed Pygments style.
