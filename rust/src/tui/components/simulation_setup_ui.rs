//! The simulation setup screen component.

use crate::tui::{
    app::{App, AppState},
    model_loader,
    simulation::SimulationState,
};
use crossterm::event::KeyCode;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

/// Handles user input on the simulation setup screen.
pub fn handle_simulation_setup_input(app: &mut App, key_code: KeyCode) {
    if let Some(setup_state) = app.simulation_setup.as_mut() {
        match key_code {
            KeyCode::Char('q') | KeyCode::Esc => {
                app.state = AppState::MainMenu;
                app.simulation_setup = None; // Clean up the state
            }
            KeyCode::Down => setup_state.next(),
            KeyCode::Up => setup_state.previous(),
            KeyCode::Tab | KeyCode::Left | KeyCode::Right => setup_state.switch_focus(),
            KeyCode::Enter => {
                if let Some(setup_state) = &app.simulation_setup {
                    if let (Some(left_index), Some(right_index)) = (
                        setup_state.left_paddle_state.selected(),
                        setup_state.right_paddle_state.selected(),
                    ) {
                        let left_model_info = &setup_state.models[left_index];
                        let right_model_info = &setup_state.models[right_index];

                        match (
                            model_loader::load_model_from_file(&left_model_info.path),
                            model_loader::load_model_from_file(&right_model_info.path),
                        ) {
                            (Ok((left_weights, left_config)), Ok((right_weights, right_config))) => {
                                app.simulation = Some(SimulationState::new(
                                    left_weights,
                                    right_weights,
                                    left_config,
                                    right_config,
                                ));
                                app.state = AppState::Simulation;
                                app.simulation_setup = None; // Clean up
                            }
                            (Err(e), _) | (_, Err(e)) => {
                                app.error_message =
                                    Some(format!("Failed to load model weights: {}", e));
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

/// Draws the UI for selecting models for a simulation.
pub fn draw_simulation_setup_ui(f: &mut Frame, app: &mut App, area: Rect) {
    if let Some(setup_state) = app.simulation_setup.as_mut() {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // For title/instructions
                Constraint::Min(0),    // For tables
            ])
            .split(area);

        let title =
            Paragraph::new("Select models for Left and Right paddles. Press Enter to start.")
                .block(
                    Block::default()
                        .title("Simulation Setup")
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded),
                )
                .alignment(Alignment::Center);
        f.render_widget(title, chunks[0]);

        let table_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        // --- Left Paddle Table ---
        let left_block = Block::default()
            .title("Left Paddle")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(if setup_state.active_table == 0 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        let header_cells = ["File", "Date Trained", "Generations"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Cyan)));
        let header = Row::new(header_cells).height(1).bottom_margin(1);

        let rows: Vec<Row> = setup_state
            .models
            .iter()
            .map(|item| {
                let date_str = item
                    .config
                    .date_trained
                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_else(|| "N/A".to_string());
                let cells = vec![
                    Cell::from(item.path.file_name().unwrap().to_str().unwrap()),
                    Cell::from(date_str),
                    Cell::from(item.config.generations.to_string()),
                ];
                Row::new(cells).height(1)
            })
            .collect();

        let table_left = Table::new(
            rows.clone(),
            &[
                Constraint::Percentage(50),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ],
        )
        .header(header.clone())
        .block(left_block)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ");

        f.render_stateful_widget(
            table_left,
            table_chunks[0],
            &mut setup_state.left_paddle_state,
        );

        // --- Right Paddle Table ---
        let right_block = Block::default()
            .title("Right Paddle")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(if setup_state.active_table == 1 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            });

        let table_right = Table::new(
            rows,
            &[
                Constraint::Percentage(50),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ],
        )
        .header(header)
        .block(right_block)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ");

        f.render_stateful_widget(
            table_right,
            table_chunks[1],
            &mut setup_state.right_paddle_state,
        );
    }
}
