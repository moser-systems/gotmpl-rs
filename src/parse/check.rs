//! Name resolution Go's parser performs while building the tree: every
//! function must be defined and every variable declared in an enclosing
//! scope. Go rejects such templates at parse time (`function "x" not
//! defined`, `undefined variable "$x"`), so `Template::parse` runs this
//! check on each parsed tree.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::SmolStr;
use super::node::{DefineNode, Expr, ListNode, Node, PipeNode, Pos};

/// The first unresolved name, by source position.
#[derive(Debug, Clone)]
pub(crate) struct Unresolved {
    pub pos: Pos,
    pub message: String,
}

/// Checks the top-level tree and every `{{define}}`/`{{block}}` body; returns
/// the unresolved name that appears first in the source.
pub(crate) fn check(
    tree: &ListNode,
    defines: &[DefineNode],
    has_func: &dyn Fn(&str) -> bool,
) -> Option<Unresolved> {
    let mut first: Option<Unresolved> = None;
    let bodies = core::iter::once(tree).chain(defines.iter().map(|d| &d.body));
    for body in bodies {
        // Each tree starts with only `$` in scope, as in Go's parser.
        let mut walker = Walker {
            vars: vec![SmolStr::new_static("$")],
            has_func,
            first: None,
        };
        walker.list(body);
        if let Some(err) = walker.first
            && first.as_ref().is_none_or(|f| err.pos.offset < f.pos.offset)
        {
            first = Some(err);
        }
    }
    first
}

struct Walker<'a> {
    vars: Vec<SmolStr>,
    has_func: &'a dyn Fn(&str) -> bool,
    first: Option<Unresolved>,
}

impl Walker<'_> {
    fn fail(&mut self, pos: Pos, message: String) {
        if self.first.is_none() {
            self.first = Some(Unresolved { pos, message });
        }
    }

    fn list(&mut self, list: &ListNode) {
        for node in &list.nodes {
            self.node(node);
        }
    }

    fn node(&mut self, node: &Node) {
        match node {
            Node::Action(action) => self.pipe(&action.pipe),
            Node::If(branch) | Node::Range(branch) | Node::With(branch) => {
                // Go pops the variables declared in the control's pipeline and
                // bodies at its {{end}}; the else body still sees them.
                let mark = self.vars.len();
                self.pipe(&branch.pipe);
                self.list(&branch.body);
                if let Some(else_body) = &branch.else_body {
                    self.list(else_body);
                }
                self.vars.truncate(mark);
            }
            Node::Template(call) => {
                if let Some(pipe) = &call.pipe {
                    self.pipe(pipe);
                }
            }
            Node::List(list) => self.list(list),
            Node::Text(_) | Node::Define(_) | Node::Break(_) | Node::Continue(_) => {}
        }
    }

    fn pipe(&mut self, pipe: &PipeNode) {
        if pipe.is_assign {
            for var in &pipe.decl {
                self.use_var(pipe.pos, var);
            }
        } else {
            // Go declares the variables before parsing the commands.
            self.vars.extend(pipe.decl.iter().cloned());
        }
        for command in &pipe.commands {
            for arg in &command.args {
                self.expr(arg);
            }
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Variable(pos, name, _) => self.use_var(*pos, name),
            Expr::Identifier(pos, name) => {
                if !(self.has_func)(name) {
                    self.fail(*pos, format!("function \"{name}\" not defined"));
                }
            }
            Expr::Pipe(_, pipe) => self.pipe(pipe),
            Expr::Chain(_, inner, _) => self.expr(inner),
            Expr::Dot(_)
            | Expr::Field(..)
            | Expr::String(..)
            | Expr::Number(..)
            | Expr::Bool(..)
            | Expr::Nil(_) => {}
        }
    }

    fn use_var(&mut self, pos: Pos, name: &SmolStr) {
        if !self.vars.iter().any(|v| v == name) {
            self.fail(pos, format!("undefined variable \"{name}\""));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::Parser;

    fn first_error(src: &str) -> Option<String> {
        let (tree, defines) = Parser::new(src, "{{", "}}").unwrap().parse().unwrap();
        check(&tree, &defines, &|name| matches!(name, "len" | "print")).map(|u| u.message)
    }

    #[test]
    fn undefined_functions_and_variables_are_reported() {
        assert_eq!(
            first_error("{{nope 1}}").as_deref(),
            Some("function \"nope\" not defined")
        );
        assert_eq!(
            first_error("{{$x}}").as_deref(),
            Some("undefined variable \"$x\"")
        );
        assert_eq!(
            first_error("{{$x = 1}}").as_deref(),
            Some("undefined variable \"$x\"")
        );
        assert_eq!(
            first_error("{{len (nope)}}").as_deref(),
            Some("function \"nope\" not defined")
        );
        assert_eq!(
            first_error("{{(nope).A}}").as_deref(),
            Some("function \"nope\" not defined")
        );
    }

    #[test]
    fn scopes_follow_go() {
        assert_eq!(
            first_error("{{$x := 1}}{{$x}}{{$x = 2}}{{$}}{{len $x}}"),
            None
        );
        assert_eq!(
            first_error("{{if true}}{{$x := 1}}{{end}}{{$x}}").as_deref(),
            Some("undefined variable \"$x\"")
        );
        assert_eq!(
            first_error("{{if $x := 1}}{{$x}}{{else}}{{$x}}{{end}}"),
            None
        );
        assert_eq!(
            first_error("{{range $i, $e := .}}{{$i}}{{$e}}{{end}}"),
            None
        );
        // Each define body starts with only `$` in scope.
        assert_eq!(
            first_error(r#"{{$x := 1}}{{define "t"}}{{$x}}{{end}}"#).as_deref(),
            Some("undefined variable \"$x\"")
        );
        assert_eq!(
            first_error(r#"{{template "t" $}}{{block "b" .}}{{print $}}{{end}}"#),
            None
        );
    }

    #[test]
    fn the_first_error_in_source_order_wins() {
        let err = first_error(r#"{{define "t"}}{{$late}}{{end}}{{$early}}"#);
        assert_eq!(err.as_deref(), Some("undefined variable \"$late\""));
    }
}
