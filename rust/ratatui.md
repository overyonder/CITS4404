# `ratatui` Cheat Sheet for Pong

`ratatui` can seem complex, but for our game, we only need to understand a few core concepts. Think of it like painting on a canvas.

### 1. The Core Setup (The Canvas and Easel)

This is the boilerplate you've already seen in `main.rs`.

*   **`Terminal`**: This is your entire screen or "canvas." You create it once at the start.
*   **`CrosstermBackend`**: This is the "easel" that holds your canvas. It's the engine that translates `ratatui`'s drawing commands into actual characters on your terminal.
*   **The Main Loop**: The heart of `ratatui` is the drawing loop. Everything happens inside this closure:
    ```rust
    terminal.draw(|frame| {
        // All your drawing code for one frame goes in here.
    })?;
    ```
    `frame` is your paintbrush and palette for a single frame.

### 2. Layouts and Rectangles (Sketching the Scene)

Before you can draw, you need to decide *where* to draw.

*   **`Rect`**: This is the most important struct. It's simply a rectangle defined by an `x`, `y`, `width`, and `height`. The entire terminal screen is one big `Rect`, which you can get with `frame.size()`.
*   **`Layout`**: This is a powerful tool for carving up one `Rect` into several smaller `Rect`s. We'll use it to divide the screen into our main play area and maybe a small area for scores.

**Example:** Splitting the screen to create a main play area.
```rust
// Inside the terminal.draw closure...
let main_layout = Layout::default()
    .direction(Direction::Vertical) // Split vertically
    .constraints([
        Constraint::Length(1),      // A 1-line-high area for scores
        Constraint::Min(0),         // The rest of the screen for the game
    ])
    .split(frame.size()); // Split the whole screen

let score_area = main_layout[0];
let game_area = main_layout[1];
```

### 3. Widgets (The Paint)

Widgets are the things you actually draw. For Pong, we only need two basic types.

*   **`Block`**: This is the most fundamental widget. It draws borders, titles, and padding. We will use it to create the boundary of our game board.
    ```rust
    let game_block = Block::default().borders(Borders::ALL).title("Pong");
    frame.render_widget(game_block, game_area); // Draw the block in the area we defined
    ```

*   **`Paragraph`**: This widget draws text. We'll use it for the scores.
    ```rust
    let score_text = format!("Score: {} - {}", game_state.score[0], game_state.score[1]);
    let score_paragraph = Paragraph::new(score_text).alignment(Alignment::Center);
    frame.render_widget(score_paragraph, score_area);
    ```

### 4. Drawing the Game Itself (Paddles and Ball)

This is the key part. `ratatui` doesn't have a "Sprite" or "GameObject" widget. We have to draw the paddles and ball ourselves by placing characters on the screen.

The easiest way is to treat them as tiny, single-character `Paragraph`s.

**How to draw the ball:**

1.  Get the ball's position from `game_state`.
2.  Convert the game coordinates (e.g., `[f64; 2]`) to terminal cell coordinates (`u16`, `u16`).
3.  Create a tiny `Rect` of size 1x1 at that exact cell.
4.  Render a `Paragraph` containing the ball character (`'●'`) into that `Rect`.

```rust
// Inside the terminal.draw closure, after drawing the main block...

// Get the inner area of our game block to draw within.
let game_area_inner = game_block.inner(game_area);

// 1. & 2. Calculate ball's position in terminal cells (example logic)
let ball_x = game_area_inner.x + (game_state.ball.position[0] as u16);
let ball_y = game_area_inner.y + (game_state.ball.position[1] as u16);

// 3. Create a 1x1 Rect for the ball
let ball_rect = Rect::new(ball_x, ball_y, 1, 1);

// 4. Draw the ball
frame.render_widget(Paragraph::new("●"), ball_rect);
```

You would do the exact same thing for the paddles, but using a `'█'` character and creating a `Rect` that is `1` wide and `PADDLE_HEIGHT` tall.
