use log::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    One,
    Two,
}

impl Player {
    pub fn next(&self) -> Player {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub x: usize,
    pub y: usize,
}

pub trait HexRepr {
    /// Called by `Hex` when `player` executes a move.
    fn place(&mut self, player: Player, position: Position);
    /// Called to undo a previous move by `player` at `position`
    fn unplace(&mut self, player: Player, position: Position);
    /// Called by the MCTS runtime before sending a sample to evaluation
    fn finalize(&mut self);
}

/// A Nop-implementation of HexRepr
impl HexRepr for () {
    fn place(&mut self, _: Player, _: Position) {}
    fn unplace(&mut self, _: Player, _: Position) {}
    fn finalize(&mut self) {}
}

const MAX_SIZE: usize = 8;

/// Represents the game of hex. Maintains an inner representation (commonly a wrapper around tf::Tensor) that is kept up-to-date with the game state.
/// Supports up to 8 x 8 boards.
/// We use bit-board version of the representation used in the Python version. I.e. something similar to this:
///                             
///                          O    <- (x, y) = (0, 0)
///                        O - O    <- (x, y) = (1, 0)
///                      O - O - O    <- (x, y) = (2, 0)
///                        O - O
///                          O
/// Except that both are presented as 'seen from the SW -> NE perspective'. In other words, player 2's perspective is flipped along the vertical (= swap x and y)
#[derive(Debug)]
pub struct Hex<Repr: HexRepr> {
    /// The size of the hex grid.
    pub size: usize,
    /// The current player
    pub current: Player,
    /// A bit-board representation of player 1's pieces
    pub player1: u64,
    /// A bit-board representation of player 2's pieces
    pub player2: u64,
    /// This is always kept in sync with the game. Commonly a reference to a mutable wrapper around a tensor that
    /// provides in-place updates, allowing it to be passed cheaply to tensorflow for inference.
    pub inner: Repr,
}

impl<Repr> Hex<Repr>
where
    Repr: HexRepr,
{
    pub fn from_grid(state: Vec<Vec<usize>>, size: usize, inner: Repr) -> Option<Self> {
        if state.len() != size {
            return None;
        }

        let mut p1 = Vec::new();
        let mut p2 = Vec::new();

        for (y, row) in state.iter().enumerate() {
            if row.len() != size {
                return None;
            }

            for x in 0..row.len() {
                match state[y][x] {
                    0 => continue,
                    1 => p1.push(Position { x, y }),
                    2 => p2.push(Position { x, y }),
                    _ => return None,
                }
            }
        }

        if !(p1.len() == p2.len() || p1.len() == p2.len() + 1) {
            return None;
        }

        let mut state = Self::empty(size, inner);

        let mut p1 = p1.into_iter();
        let mut p2 = p2.into_iter();

        loop {
            match (p1.next(), p2.next()) {
                (None, None) => break,
                (None, Some(_)) => unreachable!(),
                (Some(x), None) => state.place(x),
                (Some(x), Some(y)) => {
                    state.place(x);
                    state.place(y);
                }
            }
        }

        Some(state)
    }

    pub fn empty(size: usize, inner: Repr) -> Self {
        Hex {
            size,
            current: Player::One,
            player1: 0,
            player2: 0,
            inner,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.player1 == 0 && self.player2 == 0
    }

    pub fn minimal(&self) -> Hex<()> {
        Hex {
            inner: (),
            size: self.size,
            current: self.current,
            player1: self.player1,
            player2: self.player2,
        }
    }

    pub fn with_inner<R: HexRepr>(&self, inner: R) -> Hex<R> {
        Hex {
            size: self.size,
            current: self.current,
            player1: self.player1,
            player2: self.player2,
            inner,
        }
    }

    /// Places the piece at a given position, and advances to the next player
    /// Note: does not care for feasibility. I.e. it allows placement on already-occupied spots.
    pub fn place(&mut self, position: Position) {
        let one = self.index(position, Player::One);
        let two = self.index(position, Player::Two);
        match self.current {
            Player::One => {
                self.player1 |= 1 << one;
                //self.player2 &= !(1 << two);
            }
            Player::Two => {
                //self.player1 &= !(1 << one);
                self.player2 |= 1 << two;
            }
        }

        self.current = self.current.next();
    }

    /// Undoes a previous move at a given position
    pub fn unplace(&mut self, position: Position) {
        debug!("undoing {:?}", (position.x, position.y));
        // Since we only have two players, the next is also the previous
        let previous = self.current.next();

        match previous {
            Player::One => self.player1 &= !(1 << self.index(position, Player::One)),
            Player::Two => self.player2 &= !(1 << self.index(position, Player::Two)),
        }

        self.current = previous;
    }

    /// Converts a position `(x, y)` to the given offset in `player`'s bit board.
    fn index(&self, position: Position, player: Player) -> usize {
        match player {
            Player::One => position.x + MAX_SIZE * position.y,
            Player::Two => position.y + MAX_SIZE * position.x,
        }
    }

    /// Whether or not a given position is occupied by a given player
    pub fn is_occupied(&self, position: Position, player: Player) -> bool {
        match player {
            Player::One => self.player1 & (0b1 << self.index(position, Player::One)) != 0,
            Player::Two => self.player2 & (0b1 << self.index(position, Player::Two)) != 0,
        }
    }

    /// The game of Hex has been proven to never end in a draw. Thus, we have a winner iff the game is finished.
    pub fn winner(&self) -> Option<Player> {
        const LOWER_EIGHT_SET: u64 = 0b1111_1111;
        // Both boards have the same "perspective" as that of player 1, and can be treated equivalently
        for (board, player) in [self.player1, self.player2]
            .iter()
            .zip(&[Player::One, Player::Two])
        {
            // We will apply a BFS-type search going row-by-row.
            // We assume that all board representations are aligned on byte boundaries
            let mut reachable = [0; MAX_SIZE + 1];
            reachable[0] = board & LOWER_EIGHT_SET;
            // Note: size^2 is sufficient to ensure total exploration,
            // but it can probably be tightened quite a lot
            for _ in 0..self.size * self.size {
                for i in 1..self.size {
                    let row = (board >> MAX_SIZE * i) & LOWER_EIGHT_SET;
                    // Those that are directly reachable in row `i` form row `i - 1`
                    reachable[i] = row
                        & (reachable[i - 1] >> 1
                            | reachable[i - 1]
                            | reachable[i + 1] << 1
                            | reachable[i + 1]
                            | reachable[i] << 1
                            | reachable[i] >> 1);
                }
            }

            if reachable[self.size - 1] != 0 {
                return Some(*player);
            }
        }

        None
    }

    /// Returns all the available actions, i.e. all board positions which are currently unoccupied
    pub fn available_actions(&self) -> impl Iterator<Item = Position> + '_ {
        let n = self.size;
        (0..n).flat_map(move |x| {
            (0..n).filter_map(move |y| {
                let position = Position { x, y };

                match !self.is_occupied(position, Player::One)
                    && !self.is_occupied(position, Player::Two)
                {
                    true => Some(Position { x, y }),
                    false => None,
                }
            })
        })
    }
}
